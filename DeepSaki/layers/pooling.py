"""Collection of pooling layer operations to reduce the spatial dimensionality of a feature map."""
from enum import Enum
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional

import tensorflow as tf

class GlobalSumPooling2D(tf.keras.layers.Layer):
    """Global sum pooling operation for spatial data.

    Tips:
        Similar to tensorflow's [GlobalMaxPooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalMaxPooling2D)
        and [GlobalAveragePooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling2D)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the `GlobalSumPooling2D` object.

        Args:
            kwargs (Any): Additional key word arguments passed to the base class.
        """
        super(GlobalSumPooling2D, self).__init__(**kwargs)
        self.data_format = "channels_last"
        self.input_spec = tf.keras.layers.InputSpec(ndim=4)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> Optional[tf.TensorShape]:
        """Computes the output shape of the layer.

        This method will cause the layer's state to be built, if that has not happened before. This requires that the
        layer will later be used with inputs that match the input shape provided here.

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor to this layer.

        Returns:
            A TensorShape instance representing the shape of the layer's output Tensor.
        """
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            return tf.TensorShape([input_shape[0], input_shape[3]])
        return None

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the `GlobalSumPooling2D` layer.

        Args:
            inputs (tf.Tensor): Tensor of shape (`batch`,`height`,`width`,`channel`) or
                (`batch`,`channel`,`height`,`width`).

        Returns:
            Tensor of shape (`batch`,`channel`) where the elements are summed to reduce the axis.
        """
        return tf.reduce_sum(input_tensor=inputs, axis=[1, 2], keepdims=False)

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(GlobalSumPooling2D, self).get_config()
        config.update({"data_format": self.data_format})
        return config








class LearnedPooling(tf.keras.layers.Layer):
    """Layer that learns a pooling operation.

    Instead of using [MaxPooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPooling2D) or
    [AveragePooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D) the layer learns a
    pooling operation.

    Info:
        Under the hood this layer simply performs a non-overlapping convolution operation.
    """

    def __init__(self, pool_size: int = 2, **kwargs: Any) -> None:
        """Initializes an instance of `LearnedPooling`.

        Args:
            pool_size (int, optional): Size of the pooling window and the stride of the convolution operation. If the
                input is of shape (b, h, w, c), the output is (b,h/pool_size,w/pool_size,c). Defaults to 2.
            kwargs (Any): Additional key word arguments passed to the base class.
        """
        super(LearnedPooling, self).__init__(**kwargs)
        self.pool_size = pool_size

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer depending on the `input_shape` (output shape of the previous layer).

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor to this layer.
        """
        self.pooling = tf.keras.layers.Conv2D(
            kernel_size=self.pool_size,
            strides=self.pool_size,
            filters=input_shape[-1],
            use_bias=False,
            padding="same",
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Calls the `LearnedPooling` layer.

        Args:
            x (tf.Tensor): Tensor of shape (`batch`,`height`,`width`,`channel`).

        Returns:
            Pooled tensor of shape (`batch`,`height/pool_size`,`width/pool_size`,`channel`)
        """
        return self.pooling(x)

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(LearnedPooling, self).get_config()
        config.update({"pool_size": self.pool_size})
        return config
