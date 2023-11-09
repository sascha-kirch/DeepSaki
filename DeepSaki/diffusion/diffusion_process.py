from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf

from DeepSaki.diffusion.schedule import BetaSchedule
from DeepSaki.tensor_ops.tensor_ops import sample_array_to_tensor
from DeepSaki.types.diffusion_enums import ScheduleType
from DeepSaki.types.diffusion_enums import variance_type

# TODO: transform math into math-expressions using $...$
class GaussianDiffusionProcess:
    """Abstraction of a gausian diffusion process."""

    def __init__(self, schedule_type: ScheduleType, variance_type: variance_type, diffusion_steps: int) -> None:
        """Abstraction of a gausian diffusion process.

        Args:
            schedule_type (ScheduleType): type of beta schedule for the forward diffusion process.
            variance_type (variance_type): Type of variance estimation.
        """
        self.diffusion_steps = diffusion_steps
        self.schedule_type = schedule_type
        self.beta_schedule = BetaSchedule(schedule=self.schedule_type, timesteps=self.diffusion_steps)
        self.policy = tf.keras.mixed_precision.global_policy()
        self.variance_type = variance_type

    def draw_random_timestep(
        self,
        num: int,
    ) -> tf.Tensor:
        """Draws a random timestep from a uniform distribution.

        Args:
            num (int): Batchsize and number of samples drawn from the distribution

        Returns:
            tf.Tensor: Random timestep
        """
        return tf.random.uniform(shape=[num], minval=0, maxval=self.beta_schedule.config["timesteps"], dtype=tf.int32)

    def q_xt_given_x0_mean_var(
        self,
        x0: tf.Tensor,
        t: Union[tf.Tensor, np.ndarray],
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Forward diffusion kernel q(xt|x0).

        Args:
            x0 (tf.Tensor): Original data sample.
            t (Union[tf.Tensor, np.ndarray]): timestep

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: mean, variance and log(variance).
        """
        mean = sample_array_to_tensor(self.beta_schedule.sqrt_alpha_bar, t) * x0
        var = sample_array_to_tensor(self.beta_schedule.one_minus_alpha_bar, t)
        log_var = sample_array_to_tensor(self.beta_schedule.log_one_minus_alpha_bar, t)
        return mean, var, log_var

    def q_sample_xt(
        self,
        x0: tf.Tensor,
        t: Union[tf.Tensor, np.ndarray],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Generates a sample xt ~ q(xt|x0) at timestep `t` using the reparameterization trick.

        Args:
            x0 (tf.Tensor): Original data sample
            t (Union[tf.Tensor, np.ndarray]): timestep for which noisy sample shall be created.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Noisy sample `xt` and `noise` used to construct the sample.
        """
        noise = tf.random.normal(shape=x0.shape, dtype=self.policy.variable_dtype)
        tensor_sqrt_alpha_bar_t = sample_array_to_tensor(self.beta_schedule.sqrt_alpha_bar, t)
        tensor_sqrt_one_minus_alpha_bar_t = sample_array_to_tensor(self.beta_schedule.sqrt_one_minus_alpha_bar, t)
        xt = tensor_sqrt_alpha_bar_t * x0 + tensor_sqrt_one_minus_alpha_bar_t * noise
        return xt, noise

    def q_xtm1_given_x0_xt_mean_var(
        self,
        x0: tf.Tensor,
        xt: tf.Tensor,
        t: Union[tf.Tensor, np.ndarray],
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Tractible posterior distribution q(x(t-1)|xt,x0).

        Args:
            x0 (tf.Tensor): Original data sample
            xt (tf.Tensor): Noisy data sample at timestep t.
            t (Union[tf.Tensor, np.ndarray]): timestep.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: mean, variance and log(variance).
        """
        mean = (
            sample_array_to_tensor(self.beta_schedule.posterior_mean_coef1, t) * x0
            + sample_array_to_tensor(self.beta_schedule.posterior_mean_coef2, t) * xt
        )
        var = sample_array_to_tensor(self.beta_schedule.posterior_var, t)
        log_var = sample_array_to_tensor(self.beta_schedule.posterior_log_var_clipped, t)
        return mean, var, log_var

    def p_xtm1_given_xt_mean_var(
        self,
        xt: tf.Tensor,
        t: Union[tf.Tensor, np.ndarray],
        model_prediction: tf.Tensor,
        clip_denoised: bool = True,
        denoised_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    ) -> Dict[str, tf.Tensor]:
        """Aproximated intractible posterior distribution p(x_{t-1}|x_t).

        Real posterior distribution q(x_{t-1}|x_t) is intractible, hence it is aproximated by a neural network that
        learnes the approximated intractible posterior distribution p(x_{t-1}|x_t).

        Args:
            xt (tf.Tensor): Noisy data sample at timestep t.
            t (Union[tf.Tensor, np.ndarray]): timestep.
            model_prediction (tf.Tensor): Output of the network predicting the noise and optionally the variance.
            clip_denoised (bool, optional): If true `pred_x0` is clipped [-1,1]. Defaults to True.
            denoised_fn (Optional[Callable[[tf.Tensor], tf.Tensor]], optional): Function to call upon a prediction of
                `x0`. Defaults to None.

        Raises:
            ValueError: if provided `self.variance_type` is not specified.

        Returns:
            Dict[str, tf.Tensor]: mean, variance and log(variance) of the approximated posterior and a prediction of
                the initial x0. Keys are `mean`, `var`, `log_var` and `pred_x0`
        """
        # Variance
        if self.variance_type == variance_type.LEARNED_RANGE:
            pred_noise, pred_var = tf.split(
                model_prediction, 2, axis=-1
            )  # assumes that model has two outputs when variance is learned
            # LEARN_VARIANCE learns a range, not the final value! Was proposed by improved DDPM paper
            log_lower_bound = sample_array_to_tensor(self.beta_schedule.posterior_log_var_clipped, t)
            log_upper_bound = sample_array_to_tensor(tf.math.log(self.beta_schedule.betas), t)
            # pred_var is [-1, 1] for [log_lower_bound, log_upper_bound]. -> scaling to [0,1]
            v = (pred_var + 1) / 2
            log_var = v * log_upper_bound + (1 - v) * log_lower_bound
            var = tf.math.exp(log_var)  # = Sigma_theta from the paper
        elif self.variance_type == variance_type.LEARNED:
            pred_noise, pred_var = tf.split(
                model_prediction, 2, axis=-1
            )  # assumes that model has two outputs when variance is learned
            log_var = pred_var
            var = tf.math.exp(pred_var)  # = Sigma_theta from the paper
        elif self.variance_type == variance_type.LOWER_BOUND:
            pred_noise = model_prediction
            var = self.beta_schedule.posterior_var
            log_var = tf.math.log(self.beta_schedule.posterior_var)
        elif self.variance_type == variance_type.UPPER_BOUND:
            pred_noise = model_prediction
            var = self.beta_schedule.betas
            log_var = tf.math.log(self.beta_schedule.betas)
        else:
            raise ValueError(f"Variance type {self.variance_type} is not defined.")

        var = sample_array_to_tensor(var, t)
        log_var = sample_array_to_tensor(log_var, t)

        # predicted x0
        def _process_x0(x: tf.Tensor) -> tf.Tensor:
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return tf.clip_by_value(x, -1, 1)
            return x

        pred_x0 = _process_x0(self._predict_x0_from_eps(xt, t, pred_noise))

        # mean
        mean, _, _ = self.q_xtm1_given_x0_xt_mean_var(pred_x0, xt, t)

        return {"mean": mean, "var": var, "log_var": log_var, "pred_x0": pred_x0}

    def p_sample_xtm1_given_xt(
        self,
        xt: tf.Tensor,
        model_prediction: tf.Tensor,
        t: Union[tf.Tensor, np.ndarray],
    ) -> tf.Tensor:
        """Generates a sample x_{t-1} ~ p(x_{t-1}|xt), the approximated intractible posterior distr. at timestep `t`.

        Used by the DDPM sampler.

        Args:
            xt (tf.Tensor): Noisy data sample at timestep t.
            model_prediction (tf.Tensor): Output of the network predicting the noise and optionally the variance.
            t (Union[tf.Tensor, np.ndarray]): timestep

        Returns:
            tf.Tensor: less noisy sample `x_{t-1}`
        """
        p_xtm1_given_xt = self.p_xtm1_given_xt_mean_var(xt, t, model_prediction)

        z = tf.random.normal(shape=xt.shape)

        # only add noise if t>0
        mask = 1 if t > 0 else 0

        return p_xtm1_given_xt["mean"] + mask * tf.math.sqrt(p_xtm1_given_xt["var"]) * z

    def _predict_x0_from_eps(
        self,
        xt: tf.Tensor,
        t: Union[tf.Tensor, np.ndarray],
        eps: tf.Tensor,
    ) -> tf.Tensor:
        """Approximates the initial data sample `x0` from a noise prediction, the beta schedule and the noisy sample.

        Equation 15 DDPM paper.

        Args:
            xt (tf.Tensor): Noisy data sample at timestep t
            t (Union[tf.Tensor, np.ndarray]): timestep
            eps (tf.Tensor): Output of the network predicting the noise

        Returns:
            tf.Tensor: Approximation of x0
        """
        tensor_sqrt_recip_alpha_bar = sample_array_to_tensor(self.beta_schedule.sqrt_recip_alpha_bar, t)
        tensor_sqrt_recip_alpha_bar_minus_one = sample_array_to_tensor(
            self.beta_schedule.sqrt_recip_alpha_bar_minus_one, t
        )
        return tensor_sqrt_recip_alpha_bar * xt - tensor_sqrt_recip_alpha_bar_minus_one * eps
