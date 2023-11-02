from typing import List
from typing import Optional

import tensorflow as tf

class ComplexInitializer(tf.keras.initializers.Initializer):
    """Wraps an initializer to generate a complex-valued tensor for initialization.

    **Example:**
    ```python title="Standalone usage with single initializer for real and imaginary part."
    import DeepSaki as ds
    initializer = ds.initializers.ComplexInitializer(tf.keras.initializers.HeNormal())
    values = initializer(shape=(2, 2))
    print(values)
    # Output: <tf.Tensor: shape=(2, 2), dtype=complex64, numpy=
    #         array([[2.146195  +2.146195j  , -0.61122435-0.61122435j],
    #                [-1.9168953 -1.9168953j ,  0.7456977 +0.7456977j ]], dtype=complex64)>
    ```
    ```python title="Standalone usage with seperate initializer for real and imaginary part."
    import DeepSaki as ds
    initializer = ds.initializers.ComplexInitializer(
        initializer_real = tf.keras.initializers.HeNormal(),
        initializer_imag = tf.keras.initializers.Ones()
        )
    values = initializer(shape=(2, 2))
    print(values)
    # Output: <tf.Tensor: shape=(2, 2), dtype=complex64, numpy=
    #         array([[ 0.93472564+1.j,  1.364944  +1.j],
    #                [-0.18844277+1.j,  0.46844056+1.j]], dtype=complex64)>
    ```
    ```python title="Usage in a Keras layer"
    import DeepSaki as ds
    initializer = ds.initializers.ComplexInitializer(tf.keras.initializers.GlorotUniform())
    layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
    ```
    """

    def __init__(
        self,
        initializer_real: tf.keras.initializers.Initializer,
        initializer_imag: Optional[tf.keras.initializers.Initializer] = None,
    ) -> None:
        """Initializes the `ComplexInitializer` instance.

        Args:
            initializer_real (tf.keras.initializers.Initializer): Initializer used to initialize the real part of the
                complex tensor.
            initializer_imag (Optional[tf.keras.initializers.Initializer], optional): Initializer used to initialize
                the imaginary part of the complex tensor. If `None`, the same initializer is used as for the real part.
                Defaults to None.
        """
        self.initializer_real = initializer_real
        self.initializer_imag = initializer_imag if initializer_imag is not None else initializer_real

    def __call__(
        self,
        shape: List[int],
        dtype: tf.DType = tf.complex64,
    ) -> tf.Tensor:
        """Calls the initializer.

        Args:
            shape (List[int]): Shape of the tensor to be initialized.
            dtype (tf.DType, optional): dtype of the individual terms of the complex number. Defaults to tf.complex64.

        Returns:
            Complex-valued tensor with values drawn from the `initializer_real` and `initializer_imag`.
        """
        if dtype == tf.complex64:
            dtype = tf.float32
        elif dtype == tf.complex128:
            dtype = tf.float64
        real = self.initializer_real(shape, dtype)
        imag = self.initializer_imag(shape, dtype)
        return tf.dtypes.complex(real, imag)
