import tensorflow as tf

class NonNegative(tf.keras.constraints.Constraint):
    """Constraint that enforces positive activations."""

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        """Clips negative values of a provided weight tensor `w` to 0.0 and leaves positive values unchanged.

        Args:
            w (tf.Tensor): Tensor containing the wehights of a layer.

        Returns:
            Tensor where negative values are clipped to 0.0.
        """
        return w * tf.cast(tf.math.greater_equal(w, 0.0), w.dtype)
