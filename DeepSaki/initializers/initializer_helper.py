from typing import Callable
from typing import List

import tensorflow as tf


def make_initializer_complex(
    initializer: tf.keras.initializers.Initializer,
) -> Callable[[List[int], tf.DType], tf.complex]:
    """Returns a function that applies a given `initializer` to generate a complex-valued tensor for initialization.

    The function applies the initializer twice, once for the `real` part and once for the `imaginary` part and then
    constructs a complex-valued tensor of the provided `shape`.


    **Examples:**
    ```python
    # Standalone usage:
    import DeepSaki as dsk
    initializer = dsk.initializers.make_initializer_complex(tf.keras.initializers.GlorotUniform())
    values = initializer(shape=(2, 2))
    ```
    ```python
    # Usage in a Keras layer:
    import DeepSaki as dsk
    initializer = dsk.initializers.make_initializer_complex(tf.keras.initializers.GlorotUniform())
    layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
    ```

    Args:
        initializer (tf.keras.initializers.Initializer): Any real valued initializer object.

    Returns:
        Wrapper function with same function signature as a `tf.keras.initializers.Initializer` object.
    """

    def complex_initializer(shape: List[int], dtype: tf.DType = tf.complex64) -> tf.complex:
        """Function that applies a given `initializer` to generate a complex-valued tensor for initialization.

        Args:
            shape (List[int]): Shape of the tensor to be initialized.
            dtype (tf.DType, optional): dtype of the individual terms of the complex number. Defaults to `tf.complex64`.

        Returns:
            Complex-valued tensor with values drawn from the `initializer`, seperatly for `real` and `imaginary` part.
        """
        if dtype == tf.complex64:
            dtype = tf.float32
        elif dtype == tf.complex128:
            dtype = tf.float64
        real = initializer(shape, dtype)
        imag = initializer(shape, dtype)
        return tf.dtypes.complex(real, imag)

    return complex_initializer
