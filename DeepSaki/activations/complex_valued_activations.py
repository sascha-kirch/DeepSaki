"""Activation functions applicable to complex-valued and real-valued inputs."""

from typing import Any
from typing import Dict
from typing import Union

import tensorflow as tf

class ComplexActivation(tf.keras.layers.Layer):
    """Wrapper to apply a given `activation` to a complex input individually for the real and imaginary part.

    Inherits from:
        tf.keras.layers.Layer
    """

    def __init__(self, activation: tf.keras.layers.Layer, **kwargs: Any) -> None:
        """Initialize ComplexActivation.

        Args:
            activation (tf.keras.layers.Layer): Activation function to complexyfy.
            kwargs: keyword arguments passed to the parent class tf.keras.layers.Layer.
        """
        super(ComplexActivation, self).__init__(**kwargs)
        self.activation = activation

    def call(self, inputs: tf.Tensor) -> Union[tf.complex64, tf.complex128]:
        """Splits its intput `inputs`into a real and imaginary part, applies `activation` and constructs a complex number.

        Args:
            inputs (tf.Tensor): Input tensor to be activated. Might be a complex or real valued tensor.

        Returns:
            tf.complex64 | tf.complex128: Complex tensor with activated real and imaginary part.
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
