from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Union

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from DeepSaki.diffusion.diffusion_process import GaussianDiffusionProcess
from DeepSaki.diffusion.schedule import BetaSchedule
from DeepSaki.tensor_ops.tensor_ops import sample_array_to_tensor
from DeepSaki.types.diffusion_enums import variance_type

class SamplerResult:
    """Class holding and post processing the output of a Sampler."""

    def __init__(
        self,
        x0: tf.Tensor,
        xt_frame_list: List[tf.Tensor],
        timestep_list: List[np.ndarray],  # TODO: change to tf.Tensor
    ) -> None:
        """Initialize the `SamplerResult` instance.

        Args:
            x0 (tf.Tensor): Denoised sample.
            xt_frame_list (List[tf.Tensor]): List of tf.Tensors with the intermediate sampler outputs.
            timestep_list (List[np.ndarray]): List of timestamps.
        """
        self.x0 = x0
        # swapping shape from (frame, batch, ...) to (batch, frame, ...)
        self.xt_frames = tf.experimental.numpy.swapaxes(tf.convert_to_tensor(xt_frame_list), 0, 1)
        self.t_frames = tf.convert_to_tensor(timestep_list)


class Sampler(ABC):
    """Abstract base class for a sampler object."""

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
        diffusion_model: tf.keras.Model,
        diffusion_process: GaussianDiffusionProcess,
        diffusion_input_shape: List[int],
        save_rate: int = 20,
    ) -> None:
        """Initializes the `DDPMSample` object.

        Args:
            diffusion_model (tf.keras.Model): Model that shall be sampled from.
            diffusion_process (GaussianDiffusionProcess): The diffusion process object used for the forward diffusion.
            diffusion_input_shape (List[int]): shape of the diffusion input of the model.
            save_rate (int, optional) Rate at which images shall be stored for returning. E.g. 20 means every 20th
                image is returned. Defaults to 20.

        """
        self.diffusion_model = diffusion_model
        self.diffusion_process = diffusion_process
        self.diffusion_input_shape = diffusion_input_shape
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
    """Sampler object to use the DDIM sampling technique."""

    def __init__(
        self,
        sampling_steps: int,
        diffusion_process: GaussianDiffusionProcess,
        diffusion_model: tf.keras.Model,
        diffusion_input_shape: List[int],
    ) -> None:
        self.diffusion_process = diffusion_process
        self.diffusion_model = diffusion_model
        self.diffusion_input_shape = diffusion_input_shape
        self.sampling_steps = min(sampling_steps, self.diffusion_process.diffusion_steps)

    def sample(self, x0_condition: tf.Tensor) -> SamplerResult:
        xt_sample_list = []
        timestep_list = []
        batch_size = x0_condition.shape[0]

        # First sample is random noise
        xt_sample = tf.random.normal((batch_size, *self.diffusion_input_shape))

        # Iterate over samplesteps
        step_size = self.diffusion_process.diffusion_steps // self.sampling_steps
        for i in tqdm(range(self.diffusion_process.diffusion_steps, 0, -step_size), ncols=100, desc="DDIM sampling"):
            t = np.expand_dims(np.array(i, np.int32), 0)
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
        t: Union[tf.Tensor, np.ndarray],  # TODO: Change to tensor.
        t_prev: Union[tf.Tensor, np.ndarray],  # TODO: Change to tensor.
        beta_schedule: BetaSchedule,
        sigma_t: tf.Tensor = 0,
    ) -> tf.Tensor:
        """Sample $x_{t-1} ~ p(x_{t-1}|xt)$, the approximated posterior distr. using DDIM backward diffusion process.

        Args:
            xt (tf.Tensor): Noisy data sample at timestep t.
            model_prediction (tf.Tensor): Output of the diffusion model.
            t (Union[tf.Tensor, np.ndarray]): timestep.
            t_prev (Union[tf.Tensor, np.ndarray]): previous timestep.
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
