"""Collection of padding layer operations."""
from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple

import tensorflow as tf


class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding: Tuple[int, int] = (1, 1), **kwargs: Any) -> None:
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)

    @tf.custom_gradient
    def _padding_func(self, input_tensor: tf.Tensor) -> Tuple[tf.Tensor, Callable[[tf.Tensor], tf.Tensor]]:
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        result = tf.pad(input_tensor, padding_tensor, mode="REFLECT")

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

        return result, custom_grad

    def compute_output_shape(self, input_shape: tf.TensorShape) -> Tuple[int, int, int, int]:
        """If you are using "channels_last" configuration"""
        return (
            input_shape[0],
            input_shape[1] + 2 * self.padding[0],
            input_shape[2] + 2 * self.padding[1],
            input_shape[3],
        )

    def call(self, input_tensor: tf.Tensor) -> tf.Tensor:
        return self._padding_func(input_tensor)

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(ReflectionPadding2D, self).get_config()
        config.update({"padding": self.padding})
        return config
