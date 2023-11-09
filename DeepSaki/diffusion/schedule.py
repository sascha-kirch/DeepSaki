import math
from typing import Callable

import numpy as np
import tensorflow as tf

from DeepSaki.types.diffusion_enums import ScheduleType

class BetaSchedule:
    """Abstraction of the beta schedule modulating the timesteps in the diffusion process."""

    # Inspired by:
    # https://github.com/openai/improved-diffusion/blob/783b6740edb79fdb7d063250db2c51cc9545dcd1/improved_diffusion/gaussian_diffusion.py#L18-L62

    def __init__(
        self,
        schedule: ScheduleType = ScheduleType.LINEAR,
        timesteps: int = 1000,
        start: float = 1e-4,
        stop: float = 2e-2,
        k: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        """Abstraction of the beta schedule modulating the timesteps in the diffusion process.

        Args:
            schedule (ScheduleType, optional): Type of function the schedule shall follow. Defaults to ScheduleType.LINEAR.
            timesteps (int, optional): Number of timesteps the schedule shall have. Defaults to 1000.
            start (float, optional): Start value of the schedule. Defaults to 1e-4.
            stop (float, optional): Stop value of the schedule. Defaults to 2e-2.
            k (float, optional): k term for P2 loss scaling. Defaults to 1.0.
            gamma (float, optional): gamma term for P2 loss scaling. Defaults to 1.0.
        """
        self.config = {"schedule": schedule, "timesteps": timesteps, "start": start, "stop": stop}
        self.policy = tf.keras.mixed_precision.global_policy()
        self.betas = self._get_beta_schedule(schedule, timesteps, start, stop)

        self.alphas = 1.0 - self.betas
        self.alpha_bar = tf.math.cumprod(self.alphas, 0)
        # shifts all elements by 1 to the right and adds 1 at the first position. last element is dropped due to [:-1]
        self.alpha_bar_prev = tf.concat(
            (tf.constant(np.array([1.0]), dtype=self.policy.variable_dtype), self.alpha_bar[:-1]), axis=0
        )
        # shifts all elements by 1 to the right and adds 1 at the first position. last element is dropped due to [:-1]
        self.alpha_bar_next = tf.concat(
            (self.alpha_bar[1:], tf.constant(np.array([0.0]), dtype=self.policy.variable_dtype)), axis=0
        )

        # Pre-calculated alpha coefficients
        self.one_minus_alpha_bar = 1.0 - self.alpha_bar
        self.sqrt_alpha_bar = tf.math.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = tf.math.sqrt(self.one_minus_alpha_bar)
        self.log_one_minus_alpha_bar = tf.math.log(self.one_minus_alpha_bar)
        self.sqrt_recip_alpha_bar = tf.math.sqrt(1.0 / self.alpha_bar)
        self.sqrt_recip_alpha_bar_minus_one = tf.math.sqrt(1.0 / self.alpha_bar - 1)

        # Posterior Calculations q(x_{t-1} | xt, x0)
        self.posterior_var = self.betas * (1.0 - self.alpha_bar_prev) / self.one_minus_alpha_bar

        # [1:2] returs the element 1 in form of an tensor keeping the dimensions
        self.posterior_log_var_clipped = tf.math.log(
            tf.concat((self.posterior_var[1:2], self.posterior_var[1:]), axis=0)
        )
        # Coefficients of Equation 7 of DDPM paper
        self.posterior_mean_coef1 = self.betas * tf.math.sqrt(self.alpha_bar_prev) / self.one_minus_alpha_bar
        self.posterior_mean_coef2 = (1.0 - self.alpha_bar_prev) * tf.math.sqrt(self.alphas) / self.one_minus_alpha_bar

        # Loss coefficients P2 Paper: https://arxiv.org/abs/2204.00227
        self.lambda_t = ((1.0 - self.betas) * (1.0 - self.alpha_bar)) / self.betas
        self.SNR_t = self.alpha_bar / (1.0 - self.alpha_bar)
        self.lambda_t_tick = self.lambda_t / ((k + self.SNR_t) ** gamma)

        # used in combination with the simplified loss, since the weighting of L_simple is allready 1, meaning it has
        # been multiplied with lambda_t already.
        self.lambda_t_tick_simple = 1 / ((k + self.SNR_t) ** gamma)

    def _betas_for_alpha_bar(
        self,
        num_diffusion_timesteps: int,
        alpha_bar: Callable[[float], tf.Tensor],
        max_beta: float = 0.999,
    ) -> tf.Tensor:
        """Create a beta schedule from an alpha_bar schedule.

        Beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of (1-beta)
        over time from t = [0,1].

        Args:
            num_diffusion_timesteps (int): Number of betas to produce

            alpha_bar (Callable[[float], tf.Tensor]): Function that takes an argument t from 0 to 1 and produces the
                cumulative product of (1-beta) up to that part of the diffusion process.

            max_beta (float, optional): he maximum beta to use; use values lower than 1 to prevent singularities.
                Defaults to 0.999.

        Returns:
            tf.Tensor: Beta schedule.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return tf.constant(np.array(betas), dtype=self.policy.variable_dtype)

    def _get_beta_schedule(
        self,
        schedule: ScheduleType,
        timesteps: int,
        start: float,
        stop: float,
    ) -> tf.Tensor:
        """Returns a beta schedule.

        Args:
            schedule (ScheduleType): Type of function the schedule shall follow
            timesteps (int): Number of timesteps the schedule shall have
            start (float): Start value of the schedule
            stop (float): Stop value of the schedule

        Raises:
            ValueError: if schedule is not a valid option.

        Returns:
            tf.Tensor: Beta schedule.
        """
        if schedule == ScheduleType.LINEAR:
            betas = tf.linspace(start, stop, timesteps)
        elif schedule == ScheduleType.SIGMOID:
            betas = tf.linspace(-6, 6, timesteps)
            betas = tf.math.sigmoid(betas) * (stop - start) + start
        elif schedule == ScheduleType.COSINE:
            s = 0.008
            betas = self._betas_for_alpha_bar(
                timesteps,
                lambda t: tf.math.cos((t + s) / (1 + s) * math.pi / 2) ** 2,
            )
        else:
            raise ValueError(f"Schedule {schedule} is not defined.")

        return tf.cast(betas, dtype=self.policy.variable_dtype)
