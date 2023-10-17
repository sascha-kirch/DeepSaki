import tensorflow as tf

class NonNegative(tf.keras.constraints.Constraint):
    """constraint that enforces positive activations"""

    def __call__(self, w):
        return w * tf.cast(tf.math.greater_equal(w, 0.0), w.dtype)
