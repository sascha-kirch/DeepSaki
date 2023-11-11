from typing import Optional

import numpy as np
import tensorflow as tf

from DeepSaki.diffusion.diffusion_process import GaussianDiffusionProcess
from DeepSaki.math.statistics import calc_kl_divergence_of_univariate_normal_distribution
from DeepSaki.math.statistics import calc_log_likelihood_of_discretized_gaussian
from DeepSaki.tensor_ops.tensor_ops import sample_array_to_tensor
from DeepSaki.types.losses_enums import LossWeightType

class DiffusionLoss:
    """Object to group various loss definitions and configurations of a Diffusion model loss function."""

    def __init__(
        self,
        batch_size: int,
        diffusion_process: GaussianDiffusionProcess,
        loss_weighting_type: LossWeightType = LossWeightType.SIMPLE,
        lambda_vlb: float = 1e-3,
        return_bits: bool = True,
        global_batchsize: Optional[int] = None,
    ) -> None:
        """Initializes the `DiffusionLoss` object.

        Args:
            batch_size (int): Batch size used for training.
            diffusion_process (GaussianDiffusionProcess): The diffusion process object used for the forward diffusion.
            loss_weighting_type (LossWeightType, optional): Type of weigthing applied to the simpliefied loss L_simple.
                Defaults to LossWeightType.SIMPLE.
            lambda_vlb (float, optional): Weighting factor to controll the contribution of L_VLB in the L_hybrid.
                Defaults to 1e-3.
            return_bits (bool, optional): If true, loglikelihood is given in bits, otherwise nats. Defaults to True.
                Defaults to True.
            global_batchsize (int, optional): Batch size considering all workers running in parallel in a data parallel
                setup. If `None`, `batch_size` is set as global batch size. Defaults to None.
        """
        self.batch_size = batch_size
        self.diffusion_process = diffusion_process
        self.loss_weighting_type = loss_weighting_type
        self.lambda_vlb = lambda_vlb
        self.return_bits = return_bits
        self.global_batchsize = global_batchsize or batch_size

    def _get_loss_weighting(
        self,
        timestep: tf.Tensor,
    ) -> tf.Tensor:
        """Get the value for the loss weighting depending on the weighting_type.

        Args:
            timestep (tf.Tensor): Indicies of the timesteps used to obtain the actual value from the beta schedule.

        Raises:
            ValueError: Unsupported value for self.loss_weighting_type

        Returns:
            tf.Tensor: Loss weighting factor.
        """
        if self.loss_weighting_type == LossWeightType.SIMPLE:
            # TODO: transform into tensor
            return 1.0
        if self.loss_weighting_type == LossWeightType.P2:
            return sample_array_to_tensor(
                self.diffusion_process.beta_schedule.lambda_t_tick_simple,
                timestep,
                shape=(self.batch_size, 1, 1, 1), # TODO: shape is still very specific to 4D like data like a batched image
            )
        raise ValueError(f"Undefined loss_weighting_type provided: {self.loss_weighting_type}")

    def simple_loss(
        self,
        real: tf.Tensor,
        generated: tf.Tensor,
        timestep: tf.Tensor,
    ) -> tf.Tensor:
        """Simplified loss objective as introduced by DDPM paper.

        Args:
            real (tf.Tensor): Actual noise added to a sample in the forward diffusion.
            generated (tf.Tensor): Predicted Noise added to a sample.
            timestep (tf.Tensor): Indicies of the timesteps used to obtain the actual value from the beta schedule.

        Returns:
            Loss between real and generated data.
        """
        loss = self._get_loss_weighting(timestep) * (real - generated) ** 2

        # shape of loss: [batch, height, width, channel]
        # TODO: Depends on image like data
        loss = tf.math.reduce_mean(loss, axis=[1, 2, 3])  # mean reduce each batch individually

        # manually mean reduce by global batchsize to enable multi-worker training e.g. multi-GPU
        loss = tf.math.reduce_sum(loss)
        loss *= 1.0 / self.global_batchsize

        return loss

    def vlb_loss(
        self,
        prediction: tf.Tensor,
        x0: tf.Tensor,
        xt: tf.Tensor,
        timestep: tf.Tensor,
    ) -> tf.Tensor:
        """Calculates the Variational Lower Bound loss term for a given timestep as used in improved DDPM.

        Info:
            [Improved Denoising Diffusion Probabilistic Models, Nichol et. al., 2021](http://arxiv.org/abs/2102.09672)

        Args:
            prediction (tf.Tensor): Predicted noise of the noise prediction model.
            x0 (tf.Tensor): Unnoisy data at timestep 0.
            xt (tf.Tensor): Noisy data at timestep t obtained in the forward diffusion process.
            timestep (tf.Tensor): Indicies of the timesteps used to obtain the actual value from
                the beta schedule.

        Returns:
            Variational lower bound loss term for a given timestep
        """
        loss = self.get_vlb_loss_term(prediction, x0, xt, timestep)  # shape: (batch,)

        # shape of loss: [batch,] -> no reduce mean required as in L_simple
        loss = tf.math.reduce_sum(loss)
        loss *= 1.0 / self.global_batchsize
        return loss

    def hybrid_loss(
        self,
        real_noise: tf.Tensor,
        prediction: tf.Tensor,
        x0: tf.Tensor,
        xt: tf.Tensor,
        t: tf.Tensor,
    ) -> tf.Tensor:
        """Calculates the hybrid loss defined as the sum of the simple loss and the VLB loss.

        hybrid_loss = simple_loss + vlb_loss.

        Note:
            Expects the diffusion model to not only predict the noise but also the variance.

        Args:
            real_noise (tf.Tensor): Actual noise added to a sample in the forward diffusion.
            prediction (tf.Tensor): Predicted noise and variance of of the diffusion model.
            x0 (tf.Tensor): Unnoisy data at timestep 0.
            xt (tf.Tensor): Noisy data at timestep t obtained in the forward diffusion process.
            t (tf.Tensor): Indicies of the timesteps used to obtain the actual value from the beta schedule.

        Returns:
            Hybrid loss term.
        """
        # split prediction into noise and var
        # recombine it with applying tf.stop_gradient to the noise value
        pred_noise, pred_var = tf.split(prediction, 2, axis=-1)
        pred_stopped_noise = tf.concat((tf.stop_gradient(pred_noise), pred_var), axis=-1)
        loss_simple = self.simple_loss(real_noise, pred_noise)
        loss_vlb = self.lambda_vlb * self.vlb_loss(pred_stopped_noise, x0, xt, t)
        return loss_simple + loss_vlb

    # TODO: decouple diffusion from loss
    # in contrast to OpenAI, I don't apply the mean_flat, since I'll do reduce_mean somewhen later
    def get_vlb_loss_term(
        self,
        model_prediction: tf.Tensor,
        x0: tf.Tensor,
        xt: tf.Tensor,
        t: tf.Tensor,
    ) -> tf.Tensor:
        """Calculates the individual terms of L_VLB of a given timestep. vlb_loss = L0 + Lt + ... + LT.

        Args:
            model_prediction (tf.Tensor): Predicted noise of the noise prediction model.
            x0 (tf.Tensor): Unnoisy data at timestep 0.
            xt (tf.Tensor): Noisy data at timestep t obtained in the forward diffusion process.
            t (tf.Tensor): Indicies of the timesteps used to obtain the actual value from the beta schedule.

        Returns:
            tf.Tensor: Current VLB term Lt for a given timestep t.
        """
        q_xtm1_given_x0_xt = self.diffusion_process.q_xtm1_given_x0_xt_mean_var(x0, xt, t)
        p_xtm1_given_xt = self.diffusion_process.p_xtm1_given_xt_mean_var(xt, t, model_prediction, clip_denoised=True)

        # log_var*0.5 is the same as sqrt(var), which is the standard deviation!
        descrete_log_likelihood = -calc_log_likelihood_of_discretized_gaussian(
            x0, p_xtm1_given_xt["mean"], 0.5 * p_xtm1_given_xt["log_var"], self.return_bits
        )

        kl_divergence = calc_kl_divergence_of_univariate_normal_distribution(
            q_xtm1_given_x0_xt["mean"],
            q_xtm1_given_x0_xt["log_var"],
            p_xtm1_given_xt["mean"],
            p_xtm1_given_xt["log_var"],
            variance_is_logarithmic=True,
            return_bits=self.return_bits,
        )

        # tf.where supports batched data. if t==0, log-likelihood is returned, otherwise the KL-divergence
        return tf.where((t == 0), descrete_log_likelihood, kl_divergence)

    def get_vlb_prior(
        self,
        x0: tf.Tensor,
    ) -> tf.Tensor:
        """Get the prior KL term for the variational lower-bound.

        This term can't be optimized, as it only depends on the encoder.

        Args:
            x0 (tf.Tensor): Unnoisy data at timestep 0.

        Returns:
            tf.Tensor: Prior KL term for the variational lower-bound
        """
        t = np.expand_dims(np.array(self.diffusion_process.diffusion_steps - 1, np.int32), 0)
        q_xt_given_x0 = self.diffusion_process.q_xt_given_x0_mean_var(x0, t)
        return calc_kl_divergence_of_univariate_normal_distribution(
            mean1=q_xt_given_x0["mean"],
            var1=q_xt_given_x0["log_var"],
            mean2=0.0,
            var2=0.0,
            variance_is_logarithmic=True,
            return_bits=self.return_bits,
        )
