from typing import Any
from typing import Dict

import tensorflow as tf


class ComplexActivation(tf.keras.layers.Layer):
    """Wrapper to apply a given `activation` to a complex input individually for the real and imaginary part.

    **Example:**
    ```python hl_lines="4"
    import DeepSaki as ds
    import tensorflow as tf

    activation = tf.keras.layers.LeakyReLU(alpha=0.3)

    complex_activation = ds.activations.ComplexActivation(activation=activation)
    complex_tensor = tf.complex(real=[-1.0,1.0], imag=[-2.0,2.0])
    x = complex_activation(complex_tensor)
    print(x)
    #output: <tf.Tensor: shape=(2,), dtype=complex64, numpy=array([-0.3-0.6j,  1. +2.j ], dtype=complex64)>
    ```
    """

    def __init__(self, activation: tf.keras.layers.Layer, **kwargs: Any) -> None:
        """Initialize ComplexActivation.

        Args:
            activation (tf.keras.layers.Layer): Activation function to complexyfy.
            kwargs: keyword arguments passed to the parent class tf.keras.layers.Layer.
        """
        super(ComplexActivation, self).__init__(**kwargs)
        self.activation = activation

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Splits its intput `inputs`into a real and imaginary part, applies `activation` and constructs a complex number.

        Args:
            inputs (tf.Tensor): Input tensor to be activated. Might be a complex or real valued tensor.

        Returns:
            Complex-valued tensor with activated real and imaginary part.
        """
        real = self.activation(tf.math.real(inputs))
        imag = self.activation(tf.math.imag(inputs))
        return tf.complex(real, imag)

    def get_config(self) -> Dict[str, Any]:
        """Returns configuration of class instance.

        Returns:
            Dict[str,Any]: Dictionary containing the class' configuration.
        """
        config = super(ComplexActivation, self).get_config()
        config.update({"activation": self.activation})
        return config
