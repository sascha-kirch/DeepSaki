"""Collection of padding layer operations."""
from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple

import tensorflow as tf

class ReflectionPadding2D(tf.keras.layers.Layer):
    """Reflection Padding layer with support for TPU."""

    def __init__(self, padding: Tuple[int, int] = (1, 1), **kwargs: Any) -> None:
        """Initialize the `ReflectionPadding2D` layer.

        Args:
            padding (Tuple[int, int], optional): One-sided padding added to the `hight` and `width` to an input tensor
                of shape (batch, height, width, channel)respectively. Defaults to (1, 1).
        """
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)

    @tf.custom_gradient
    def _padding_func(self, input_tensor: tf.Tensor) -> Tuple[tf.Tensor, Callable[[tf.Tensor], tf.Tensor]]:
        """Calculates results for forward prop and provides a function to calc. gradients for backward propagation.

        Args:
            input_tensor (tf.Tensor): Tensor of shape `(batch, height, width, channel)`.

        Returns:
            padded_tensor: Padded tensor of shape `(batch, height+2*padding, width+2*padding, channel)`.
            custom_grad: Function returning the gradients during backprop.
        """
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        padded_tensor = tf.pad(input_tensor, padding_tensor, mode="REFLECT")

        # upstream gradient is the chainrule of all previous gradients!
        def custom_grad(upstream: tf.Tensor) -> tf.Tensor:
            # The gradients that represent the padding are cut, since they are not relevant!
            return tf.image.crop_to_bounding_box(
                image=upstream,
                offset_height=0,
                offset_width=0,
                target_height=upstream.shape[1] - 2 * padding_height,
                target_width=upstream.shape[2] - 2 * padding_width,
            )

        return padded_tensor, custom_grad

    def compute_output_shape(self, input_shape: tf.TensorShape) -> Tuple[int, int, int, int]:
        """Calculates the expected output shape after calling the layer.

        Assumes "channels_last" configuration.

        Args:
            input_shape (tf.TensorShape): Shape of the input data to the layer.

        Returns:
            output_shape: expected output shape of the layer.
        """
        return (
            input_shape[0],
            input_shape[1] + 2 * self.padding[0],
            input_shape[2] + 2 * self.padding[1],
            input_shape[3],
        )

    def call(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """Calls the `ReflectionPadding2D` layer.

        Args:
            input_tensor (tf.Tensor): Tensor of shape `(batch, height, width, channel)`.

        Returns:
            padded tensor of shape `(batch, height+2*padding, width+2*padding, channel)`.
        """
        return self._padding_func(input_tensor)

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(ReflectionPadding2D, self).get_config()
        config.update({"padding": self.padding})
        return config
