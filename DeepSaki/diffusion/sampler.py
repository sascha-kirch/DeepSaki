from abc import ABC
from abc import abstractmethod
from typing import List

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from DeepSaki.diffusion.schedule import BetaSchedule
from DeepSaki.frameworks.diffusion_model import DiffusionModel
from DeepSaki.tensor_ops.tensor_ops import sample_array_to_tensor
from DeepSaki.types.diffusion_enums import variance_type

class SamplerResult:
    """Class holding and post processing the output of a Sampler."""

    def __init__(
        self,
        x0: tf.Tensor,
        xt_frame_list: List[tf.Tensor],
        timestep_list: List[tf.Tensor],
    ) -> None:
        """Initialize the `SamplerResult` instance.

        Args:
            x0 (tf.Tensor): Denoised sample.
            xt_frame_list (List[tf.Tensor]): List of tf.Tensors with the intermediate sampler outputs.
            timestep_list (List[tf.Tensor]): List of timestamps.
        """
        self.x0 = x0
        # swapping shape from (frame, batch, ...) to (batch, frame, ...)
        self.xt_frames = tf.experimental.numpy.swapaxes(tf.convert_to_tensor(xt_frame_list), 0, 1)
        self.t_frames = tf.convert_to_tensor(timestep_list)


class Sampler(ABC):
    """Abstract base class for a sampler object.

    can be overridden to develop a custom Sampler.
    """

    @abstractmethod
    def sample(self, x0_condition: tf.Tensor) -> SamplerResult:
        """Abstract method to sample a diffusion model.

        Args:
            x0_condition (tf.Tensor): Condition input for the conditional diffusion model.
        """
        ...


# TODO: add some math to the docstring
class DDPMSampler(Sampler):
    """Sampler object to use the DDPM sampling technique."""

    def __init__(
        self,
        diffusion_model: DiffusionModel,
        save_rate: int = 20,
    ) -> None:
        """Initializes the `DDPMSample` object.

        Args:
            diffusion_model (DiffusionModel): Diffusion model that shall be sampled from.
            save_rate (int, optional): Rate at which images shall be stored for returning. E.g. 20 means every 20th
                image is returned. Defaults to 20.

        """
        self.diffusion_model = diffusion_model
        self.diffusion_process = diffusion_model.diffusion_process
        self.diffusion_input_shape = diffusion_model.diffusion_input_shape
        self.save_rate = save_rate
        self.sampling_steps = self.diffusion_process.diffusion_steps

    def sample(self, x0_condition: tf.Tensor) -> SamplerResult:
        """Sample the diffusion model using DDPM Sampler.

        Args:
            x0_condition (tf.Tensor): Condition input for the conditional diffusion model.

        Returns:
            SamplerResult: Object containing the denoised sample `x0`, a tensor of intermediate results `xt_frames` and
                a tensor with the corresponding timesteps `t_frames`.
        """
        xt_sample_list = []
        timestep_list = []
        batch_size = x0_condition.shape[0]

        # First sample is random noise
        xt_sample = tf.random.normal((batch_size, *self.diffusion_input_shape))

        for i in tqdm(range(self.sampling_steps, 0, -1), ncols=100, desc="DDPM sampling"):
            t = np.expand_dims(np.array(i, np.int32), 0)
            model_prediction = self.diffusion_model(x0_condition, xt_sample, t, training=False)
            xt_sample = self.diffusion_process.p_sample_xtm1_given_xt(xt_sample, model_prediction, t)

            if i % self.save_rate == 0 or i == self.sampling_steps or i < 8:
                xt_sample_list.append(xt_sample)
                timestep_list.append(i)

        return SamplerResult(x0=xt_sample, xt_frame_list=xt_sample_list, timestep_list=timestep_list)


class DDIMSampler(Sampler):
    r"""Sampler object to use the DDIM sampling technique.

    DDIM accelerates the sampling process by skipping timesteps in the reverse diffusion process.
    $$
        x_{t-1} = \sqrt{\alpha_{t-1}}(\frac{x_t - \sqrt{1-\alpha_t}\epsilon_{\theta}(x_t)}{\sqrt{\alpha_t}})
                + \sqrt{1- \alpha_{t-1} - \sigma_t^2} \epsilon_{\theta}(x_t)
                + \sigma_t \epsilon_t
    $$

    If $\sigma_t = 0$, the DDIM sampler becomes deterministic.

    Note:
        DDIM paper uses $\alpha_t$ to reffer to $\bar{\alpha}_t$ from the DDPM paper.

    Info:
        DDIM was introduced in [Denoising Diffusion Implicit Models, Song et. al., 2020](https://arxiv.org/abs/2010.02502)
    """

    def __init__(
        self,
        sampling_steps: int,
        diffusion_model: DiffusionModel,
    ) -> None:
        """Initializes the `DDIMSample` object.

        Args:
            sampling_steps (int): Number of steps to divide the reverse diffusion process into.
            diffusion_model (DiffusionModel): Diffusion model that shall be sampled from.
        """
        self.diffusion_process = diffusion_model.diffusion_process
        self.diffusion_model = diffusion_model
        self.diffusion_input_shape = diffusion_model.diffusion_input_shape
        self.sampling_steps = min(sampling_steps, self.diffusion_process.diffusion_steps)

    def sample(self, x0_condition: tf.Tensor) -> SamplerResult:
        """Sample the diffusion model using DDPM Sampler.

        Args:
            x0_condition (tf.Tensor): Condition input for the conditional diffusion model.

        Returns:
            SamplerResult: Object containing the denoised sample `x0`, a tensor of intermediate results `xt_frames` and
                a tensor with the corresponding timesteps `t_frames`.
        """
        xt_sample_list = []
        timestep_list = []
        batch_size = x0_condition.shape[0]

        # First sample is random noise
        xt_sample = tf.random.normal((batch_size, *self.diffusion_input_shape))

        # Iterate over samplesteps
        step_size = self.diffusion_process.diffusion_steps // self.sampling_steps
        for i in tqdm(range(self.diffusion_process.diffusion_steps, 0, -step_size), ncols=100, desc="DDIM sampling"):
            t = np.expand_dims(np.array(i, np.int32), 0)
            # TODO: to be indipendent from the condition input etc. I could re-write the models to take 1 input (list of all other inputs) and 1 timestep input.
            model_prediction = self.diffusion_model(x0_condition, xt_sample, t, training=False)
            xt_sample = self._ddim_sample(
                xt_sample, model_prediction, t, t - step_size, self.diffusion_process.beta_schedule
            )

            xt_sample_list.append(xt_sample)
            timestep_list.append(i)

        return SamplerResult(x0=xt_sample, xt_frame_list=xt_sample_list, timestep_list=timestep_list)

    def _ddim_sample(
        self,
        xt: tf.Tensor,
        model_prediction: tf.Tensor,
        t: tf.Tensor,
        t_prev: tf.Tensor,
        beta_schedule: BetaSchedule,
        sigma_t: tf.Tensor = 0,
    ) -> tf.Tensor:
        r"""Sample $x_{t-1} \sim p(x_{t-1}|x_t)$, the approximated posterior distr. using DDIM backward diffusion process.

        Args:
            xt (tf.Tensor): Noisy data sample at timestep t.
            model_prediction (tf.Tensor): Output of the diffusion model.
            t (tf.Tensor): timestep.
            t_prev (tf.Tensor): previous timestep.
            beta_schedule (BetaSchedule): beta schedule of the diffusion process.
            sigma_t (tf.Tensor): noise tensor. Defaults to 0.

        Returns:
            tf.Tensor: less noisy sample $x_{t-1}$
        """
        if self.diffusion_process.variance_type in [variance_type.LEARNED, variance_type.LEARNED_RANGE]:
            pred_noise, _ = tf.split(model_prediction, 2, axis=-1)
        else:
            pred_noise = model_prediction

        sqrt_one_minus_alpha_bar = sample_array_to_tensor(beta_schedule.sqrt_one_minus_alpha_bar, t)
        sqrt_alpha_bar = sample_array_to_tensor(beta_schedule.sqrt_alpha_bar, t)
        alpha_bar_prev = sample_array_to_tensor(beta_schedule.alpha_bar, t_prev)

        pred = tf.math.sqrt(alpha_bar_prev) * (xt - (sqrt_one_minus_alpha_bar) * pred_noise) / sqrt_alpha_bar

        pred = pred + tf.math.sqrt(1 - alpha_bar_prev - (sigma_t**2)) * pred_noise
        noise = tf.random.normal(shape=xt.shape)
        mask = 0 if t == 0 else 1
        return pred + mask * sigma_t * noise
