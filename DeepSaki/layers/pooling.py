"""Collection of pooling layer operations to reduce the spatial dimensionality of a feature map."""
from enum import Enum
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional

import tensorflow as tf


class FrequencyFilter(Enum):
    """`Enum` used to define valid filters for `rFFT2DFilter`.

    Attributes:
        LOW_PASS (int): Indicates that low frequency components shall be kept and high frequency components shall be
            filtered.
        HIGH_PASS (int): Indicates that high frequency components shall be kept and low frequency components shall be
            filtered.
    """

    LOW_PASS = 1
    HIGH_PASS = 2


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


class rFFT2DFilter(tf.keras.layers.Layer):
    """Low or high pass filtering by truncating higher or lower frequencies in the frequency domain.

    Layer input is asumed to be in spatial domain. It is transformed into the frequency domain applying a 2D real FFT.
    Then the center-crop operation is performed and depending of the shift, either low or high frequencies are removed.
    Afterwards, the cropped region is zero padded and then the inverse real 2D FFT is calculated to transform back into
    the spatial domain.
    """

    def __init__(
        self,
        is_channel_first: bool = False,
        filter_type: Literal[FrequencyFilter.LOW_PASS, FrequencyFilter.HIGH_PASS] = FrequencyFilter.LOW_PASS,
        **kwargs: Any,
    ) -> None:
        """Initialize the `rFFT2DFilter` object.

        Args:
            is_channel_first (bool, optional): If True, input shape is assumed to be (`batch`,`channel`,`height`,`width`).
                If False, input shape is assumed to be (`batch`,`height`,`width`,`channel`). Defaults to False.
            filter_type (Literal[Frequency_Filter.LOW_PASS,Frequency_Filter.HIGH_PASS], optional): If
                `Frequency_Filter.LOW_PASS`, high frequency values are truncated, if `Frequency_Filter.HIGH_PASS`, low
                frequencies are truncated. Defaults to `Frequency_Filter.LOW_PASS`.
            kwargs (Any): Additional key word arguments passed to the base class.
        """
        super(rFFT2DFilter, self).__init__(**kwargs)
        self.is_channel_first = is_channel_first
        self.filter_type = filter_type

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer depending on the `input_shape` (output shape of the previous layer).

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor to this layer.
        """
        super(rFFT2DFilter, self).build(input_shape)
        if self.is_channel_first:
            batch_size, inp_filter, inp_height, inp_width = input_shape
        else:
            batch_size, inp_height, inp_width, inp_filter = input_shape
        self.offset_height = inp_height // 2
        self.offset_width = 0
        self.target_height = inp_height // 2
        self.target_width = int(
            inp_width / 4 + 1
        )  # 1/4 because real spectrum has allready half width and filter only applies to positive frequencies in width

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the `rFFT2DFilter` layer.

        Args:
            inputs (tf.Tensor): Tensor of shape (`batch`,`height`,`width`,`channel`) or
                (`batch`,`channel`,`height`,`width`).

        Raises:
            ValueError: If Layer has not been built.

        Returns:
            Filtered tensor with shape (`batch`,`channel`,`height`,`width`).
        """
        if not self.built:
            raise ValueError("This model has not yet been built.")

        if not self.is_channel_first:  # layer assumes channel first due to FFT
            inputs = tf.einsum("bhwc->bchw", inputs)

        inputs_f_domain = tf.signal.rfft2d(inputs)

        if self.filter_type == FrequencyFilter.LOW_PASS:
            inputs_f_domain = tf.signal.fftshift(
                inputs_f_domain, axes=[-2]
            )  # shift frequencies to be able to crop in center
        shape = tf.shape(inputs_f_domain)
        outputs_f_domain = tf.slice(
            inputs_f_domain,
            begin=[0, 0, self.offset_height, self.offset_width],
            size=[shape[0], shape[1], self.target_height, self.target_width],
        )  # Tf.slice instead of tf.image.crop, because the latter assumes channel last
        if self.filter_type == FrequencyFilter.LOW_PASS:
            outputs_f_domain = tf.signal.ifftshift(outputs_f_domain, axes=[-2])  # reverse shift
        outputs = tf.signal.irfft2d(outputs_f_domain)

        # reverse to previous channel config!
        if not self.is_channel_first:
            outputs = tf.einsum("bchw->bhwc", outputs)
        return outputs

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(rFFT2DFilter, self).get_config()
        config.update({"is_channel_first": self.is_channel_first, "filter_type": self.filter_type})
        return config


class FourierPooling2D(tf.keras.layers.Layer):
    """Pooling in frequency domain by truncating high frequencies using a center crop operation.

    Layer input is asumed to be in frequency domain and shifted, such that the center frequency is in the center of
    the grid.

    If this is the case, the center represents the frequency of 0Hz (hence an offset). The further away from the center
    the higher the frequency component. Center cropping removes high frequency components, hence can be seen as a low
    pass filter

    """

    def __init__(self, is_channel_first: bool = False, **kwargs: Any) -> None:
        """Initializes an instance of `FourierPooling2D`.

        Args:
            is_channel_first (bool, optional): If True, input shape is assumed to be (`batch`,`channel`,`height`,`width`).
                If False, input shape is assumed to be (`batch`,`height`,`width`,`channel`). Defaults to False.
            kwargs (Any): Additional key word arguments passed to the base class.
        """
        super(FourierPooling2D, self).__init__(**kwargs)
        self.is_channel_first = is_channel_first

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the `FourierPooling2D` layer.

        Args:
            inputs (tf.Tensor): Tensor of shape (`batch`,`height`,`width`,`channel`) or
                (`batch`,`channel`,`height`,`width`). Tensor is asumed to be in frequency domain of type `tf.complex64`
            or `tf.complex128`.

        Returns:
            Pooled tensor of shape (`batch`,`channel`,`height/2`,`width/2`) or (`batch`,`height/2`,`width/2`,`channel`)
        """
        if self.is_channel_first:
            inputs = tf.einsum("bchw->bhwc", inputs)

        outputs = tf.image.central_crop(inputs, 0.5)  # assumes channel last

        # reverse to previous channel config!
        if self.is_channel_first:
            outputs = tf.einsum("bhwc->bchw", outputs)
        return outputs

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(FourierPooling2D, self).get_config()
        config.update({"is_channel_first": self.is_channel_first})
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
