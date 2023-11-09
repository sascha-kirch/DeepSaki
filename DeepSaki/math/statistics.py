
import numpy as np
import tensorflow as tf

def calc_approximated_standard_normal_cdf(x: tf.Tensor) -> tf.Tensor:
    """A fast approximation of the cumulative distribution function of the standard normal.

    Args:
        x (tf.Tensor): Standard normal distribution.

    Returns:
        tf.Tensor: Approximation of the cumulative distribution function of the standard normal.
    """
    return 0.5 * (1.0 + tf.math.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * tf.math.pow(x, 3))))


def calc_log_likelihood_of_discretized_gaussian(
    xt: tf.Tensor,
    mean: tf.Tensor,
    log_std: tf.Tensor,
    return_bits: bool = True,
) -> tf.Tensor:
    """Calculates the Log Likelihood of a discretized Gaussian.

    Args:
        xt (tf.Tensor): Sample x at timestep t.
        mean (tf.Tensor): Mean value of the sample x at timestep t.
        log_std (tf.Tensor): Standard deviation in logarithmic scale of the sample x at timestep t.
        return_bits (bool, optional): If true, loglikelihood is given in bits, otherwise nats. Defaults to True.

    Returns:
        tf.Tensor: Log likelihood of a discretized Gaussian in either nats or bits.
    """
    # 1. Obtain the standard normal distribution by subtracting the mean and deviding by the standard normal distribution (mean = 0, std = 1)
    # 1.1 center xt arround 0
    centered_xt = xt - mean

    # 1.2. calc inverse of the standard deviation: 1/std. Remember log_scale and log arithmetic
    inv_std = tf.math.exp(-log_std)

    # 1.3. calculating the standard normal distribution considering the discretizing error by adding/subtracting a signle bit to the cented image and scaling it by the inverse standard deviation
    plus_in = inv_std * (centered_xt + 1 / 255.0)
    min_in = inv_std * (centered_xt - 1 / 255.0)

    # 2. calculate the CDF for the standard normal distributions discretization error
    cdf_plus = calc_approximated_standard_normal_cdf(plus_in)
    cdf_min = calc_approximated_standard_normal_cdf(min_in)
    cdf_delta = cdf_plus - cdf_min

    # 3 clip the CDFs to a minimum value and maximum value and calculate its log
    log_cdf_plus = tf.math.log(tf.clip_by_value(cdf_plus, clip_value_min=1e-12, clip_value_max=tf.float32.max))
    log_one_minus_cdf_min = tf.math.log(
        tf.clip_by_value((1 - cdf_min), clip_value_min=1e-12, clip_value_max=tf.float32.max)
    )
    log_cdf_delta = tf.math.log(tf.clip_by_value(cdf_delta, clip_value_min=1e-12, clip_value_max=tf.float32.max))

    # 4. calculate the log probs.
    # if (xt < -0.999) use cdf_log_plus.
    # if (-0.999 < xt < 0.999) use log_cdf_delta
    # if (xt > 0.999) use log_one_minus_cdf_min
    log_probs = tf.where(xt < -0.999, log_cdf_plus, tf.where(xt > 0.999, log_one_minus_cdf_min, log_cdf_delta))

    # reduce all non-batch dimensions
    # log_probs = tf.math.reduce_mean(log_probs,axis = list(range(1,len(log_probs.shape))))
    shape = tf.shape(log_probs)
    axis_to_reduce = list(range(1, len(shape)))
    log_probs = tf.math.reduce_mean(log_probs, axis=axis_to_reduce)

    if return_bits:
        log_probs = log_probs / tf.math.log(2.0)

    return log_probs


def calc_kl_divergence_of_univariate_normal_distribution(
    mean1: tf.Tensor,
    var1: tf.Tensor,
    mean2: tf.Tensor,
    var2: tf.Tensor,
    variance_is_logarithmic: bool = True,
    return_bits: bool = True,
) -> tf.Tensor:
    """Calculates the Kullback Leibler Divergence between two univariate Normal distributions.

    Args:
        mean1 (tf.Tensor): Mean of the first distribution.
        var1 (tf.Tensor): Variance of the first distribution.
        mean2 (tf.Tensor): Mean value of the second distribution.
        var2 (tf.Tensor): Mean value of the first distribution.
        variance_is_logarithmic (bool, optional): If true, indicates that provided variances are in logaritmic scale.
            Otherwise they are not. Defaults to True.
        return_bits (bool, optional): If true, loglikelihood is given in bits, otherwise nats. Defaults to True.

    Returns:
        tf.Tensor: Kullback Leibler Divergence between two univariate Normal distributions.
    """
    if variance_is_logarithmic:
        kl_divergence = 0.5 * (
            var2 - var1 + tf.math.exp(var1 - var2) + ((mean1 - mean2) ** 2) * tf.math.exp(-var2) - 1.0
        )
    else:
        kl_divergence = tf.math.log((var2 / var1) ** 0.5) + (var1 + ((mean1 - mean2) ** 2)) / (2 * var2) - 0.5

    # reduce all non-batch dimensions
    shape = tf.shape(kl_divergence)
    axis_to_reduce = list(range(1, len(shape)))
    kl_divergence = tf.math.reduce_mean(kl_divergence, axis=axis_to_reduce)

    if return_bits:
        kl_divergence = kl_divergence / tf.math.log(2.0)

    return kl_divergence
