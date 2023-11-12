import tensorflow as tf


def manually_reduce_loss(loss: tf.Tensor, global_batch_size: int) -> tf.Tensor:
    """Manually reduce a loss tensor.

    Usually tf.keras.losses.Loss reduces a loss with a mean function over the batch axis. If training in a data-parallel
    framework, one batch is distributed over multiple workers, meaning a single batch is divided in sub-batches. Each
    sub-batch calls the train_step function where losses an gradients are calculated. The results from all workers are
    gathered and summed up. If the loss would be reduced by a mean over the batch_axis, the result would be incorrect,
    since the true batch size, the global batch size, is larger, i.e.:
    $$
    batchsize_{global} = batchsize_{local} num_workers
    $$

    Args:
        loss (tf.Tensor): un-reduced loss tensor of shape [batch, ...]
        global_batch_size (int): Global batch size used for training, considering all workers training in parallel in a
            data parallel training framework.

    Returns:
        Reduced loss tensor of shape ().
    """
    # important: since batch is always expected at index 0 and different data types have different dimensions,
    # we first take care of the batch axis and then simply take the mean over the rest.
    loss = tf.math.reduce_sum(loss, axis=0) * (1.0 / global_batch_size)
    return tf.math.reduce_mean(loss)
