from typing import Union

import numpy as np
import tensorflow as tf

from DeepSaki.diffusion.diffusion_process import GaussianDiffusionProcess
from DeepSaki.math.statistics import calc_kl_divergence_of_univariate_normal_distribution
from DeepSaki.math.statistics import calc_log_likelihood_of_discretized_gaussian

# TODO: Class or func?
def L_simple(
    real: tf.Tensor,
    generated: tf.Tensor,
    global_batchsize: int,
    scaling: float = 1.0,
) -> tf.Tensor:
    """Simplified loss objective as introduced by DDPM paper.

    Args:
        real (tf.Tensor): Actual noise added to a sample in the forward diffusion.
        generated (tf.Tensor): Predicted Noise added to a sample.
        global_batchsize (int): Batch size considering all workers running in parallel in a data parallel setup.
        scaling (float, optional): Scales the loss. If 1 results in the L_simple. Defaults to 1.0.

    Returns:
        tf.Tensor: Loss between real and generated data.
    """
    loss = scaling * (real - generated) ** 2

    # shape of loss: [batch, height, width, channel]
    loss = tf.math.reduce_mean(loss, axis=[1, 2, 3])  # mean reduce each batch individually

    # manually mean reduce by global batchsize to enable multi-worker training e.g. multi-GPU
    loss = tf.math.reduce_sum(loss)
    loss *= 1.0 / global_batchsize

    return loss


def L_VLB(
    prediction: tf.Tensor,
    batched_x0_depth: tf.Tensor,
    batched_xt_depth: tf.Tensor,
    timestep_values: Union[tf.Tensor, np.ndarray],
    diffusion: GaussianDiffusionProcess,
    global_batchsize: int,
) -> tf.Tensor:
    """Calculates the Variational Lower Bound loss term for a given timestep as used in improved DDPM.

    L = Lsimple + L_VLB.

    Args:
        prediction (tf.Tensor): Predicted noise of the noise prediction model.
        batched_x0_depth (tf.Tensor): Unnoisy data at timestep 0.
        batched_xt_depth (tf.Tensor): Noisy data at timestep t obtained in the forward diffusion process.
        timestep_values (Union[tf.Tensor, np.ndarray]): Indicies of the timesteps used to obtain the actual value from
            the beta schedule.
        diffusion (GaussianDiffusion): Abstraction of the diffusion process.
        global_batchsize (int): Batch size considering all workers running in parallel in a data parallel setup.

    Returns:
        tf.Tensor: Variational lower bound loss term for a given timestep
    """
    loss = Get_L_VLB_Term(
        prediction, batched_x0_depth, batched_xt_depth, timestep_values, diffusion, clip_denoised=False
    )  # shape: (batch,)

    # shape of loss: [batch,] -> no reduce mean required as in L_simple
    loss = tf.math.reduce_sum(loss)
    loss *= 1.0 / global_batchsize
    return loss


# in contrast to OpenAI, I don't apply the mean_flat, since I'll do reduce_mean somewhen later
def Get_L_VLB_Term(
    model_prediction: tf.Tensor,
    x0: tf.Tensor,
    xt: tf.Tensor,
    t: Union[tf.Tensor, np.ndarray],
    diffusion: GaussianDiffusionProcess,
    clip_denoised: bool = True,
    return_bits: bool = True,
) -> tf.Tensor:
    """Calculates the individual terms of L_VLB of a given timestep. LVLB = L0 + Lt + ... + LT.

    Args:
        model_prediction (tf.Tensor): Predicted noise of the noise prediction model.
        x0 (tf.Tensor): Unnoisy data at timestep 0.
        xt (tf.Tensor): Noisy data at timestep t obtained in the forward diffusion process.
        t (Union[tf.Tensor, np.ndarray]): Indicies of the timesteps used to obtain the actual value from the beta schedule.
        diffusion (GaussianDiffusion): Abstraction of the diffusion process.
        clip_denoised (bool, optional): _description_. Defaults to True.
        return_bits (bool, optional): If true, loglikelihood is given in bits, otherwise nats. Defaults to True.

    Returns:
        tf.Tensor: Current VLB term Lt for a given timestep t.
    """
    true_mean, _, true_log_var_clipped = diffusion.q_xtm1_given_x0_xt_mean_var(x0, xt, t)
    p_xtm1_given_xt = diffusion.p_xtm1_given_xt_mean_var(xt, t, model_prediction, clip_denoised=clip_denoised)

    # log_var*0.5 is the same as sqrt(var), which is the standard deviation!
    dg_ll = -calc_log_likelihood_of_discretized_gaussian(
        x0, p_xtm1_given_xt["mean"], 0.5 * p_xtm1_given_xt["log_var"], return_bits
    )

    kl = calc_kl_divergence_of_univariate_normal_distribution(
        true_mean,
        true_log_var_clipped,
        p_xtm1_given_xt["mean"],
        p_xtm1_given_xt["log_var"],
        variance_is_logarithmic=True,
        return_bits=return_bits,
    )

    # tf.where supports batched data. if t==0, log-likelihood is returned, otherwise the KL-divergence
    return tf.where((t == 0), dg_ll, kl)


def Get_VLB_prior(
    x0: tf.Tensor,
    diffusion: GaussianDiffusionProcess,
    return_bits: bool = True,
) -> tf.Tensor:
    """Get the prior KL term for the variational lower-bound.

    This term can't be optimized, as it only depends on the encoder.

    Args:
        x0 (tf.Tensor): Unnoisy data at timestep 0.
        diffusion (GaussianDiffusion): Abstraction of the diffusion process.
        return_bits (bool, optional): If true, loglikelihood is given in bits, otherwise nats. Defaults to True.

    Returns:
        tf.Tensor: Prior KL term for the variational lower-bound
    """
    t = np.expand_dims(np.array(diffusion.diffusion_steps - 1, np.int32), 0)
    qt_mean, _, qt_log_variance = diffusion.q_xt_given_x0_mean_var(x0, t)
    return calc_kl_divergence_of_univariate_normal_distribution(
        mean1=qt_mean, var1=qt_log_variance, mean2=0.0, var2=0.0, variance_is_logarithmic=True, return_bits=return_bits
    )
