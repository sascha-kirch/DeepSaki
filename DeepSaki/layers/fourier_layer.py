from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import tensorflow as tf

from DeepSaki.initializers.initializer_helper import make_initializer_complex

class MultiplicationType(Enum):
    """`Enum` used to define how two matrices shall be multiplied.

    Attributes:
        ELEMENT_WISE (int): Indicates to apply an element-wise multiplication of 2 tensors.
        MATRIX_PRODUCT (int): Indicates to apply a matrix-product between 2 tensors.
    """

    ELEMENT_WISE = 1
    MATRIX_PRODUCT = 2


class FourierConvolution2D(tf.keras.layers.Layer):
    """Performs a convolution by transforming into fourier domain. Layer input is asumed to be in spatial domain."""

    def __init__(
        self,
        filters: int,
        kernels: Optional[Tuple[int, int]] = None,
        use_bias: bool = True,
        kernel_initializer: Optional[tf.keras.initializers.Initializer] = None,
        bias_initializer: Optional[tf.keras.initializers.Initializer] = None,
        is_channel_first: bool = False,
        apply_conjugate: bool = False,
        pad_to_power_2: bool = True,
        method: MultiplicationType = MultiplicationType.ELEMENT_WISE,
        **kwargs: Any,
    ) -> None:
        """Initialize the `FourierConvolution2D` object.

        Args:
            filters (int): Number of individual filters.
            kernels (Optional[Tuple[int, int]], optional): Kernel of the spatial convolution. Expected input
                `[height,width]`. If `None`, kernel size is set to the input height and width. Defaults to `None`.
            use_bias (bool, optional): Whether or not to us bias weights. Defaults to `True`.
            kernel_initializer (tf.keras.initializers.Initializer, optional): Initializer to initialize the kernels of
                the convolution layer. Defaults to `None`.
            bias_initializer (tf.keras.initializers.Initializer, optional): Initializer to initialize the bias weights
                of the convolution layer. Defaults to `None`.
            is_channel_first (bool, optional): Set true if input shape is (b,c,h,w) and false if input shape is
                (b,h,w,c). Defaults to `False`.
            apply_conjugate (bool, optional): If true, the kernels are conjugated. If so, a multiplication in the
                frequency domain corresponds to a cross correlation in the spatial domain, which is actually what a
                convolution layer is doing. Defaults to `False`.
            pad_to_power_2 (bool, optional): If true, input tensor is padded. FFT algorithm runs faster for lengths of
                power of two. Defaults to `True`.
            method (MultiplicationType, optional): Type of multiplication of the input and the weights:
                [`MultiplicationType.ELEMENT_WISE` | `MultiplicationType.MATRIX_PRODUCT`]. Defaults to
                `MultiplicationType.ELEMENT_WISE`.
            kwargs (Any): Additional key word arguments passed to the base class.
        """
        super(FourierConvolution2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernels = kernels
        self.use_bias = use_bias
        self.kernel_initializer = (
            tf.keras.initializers.RandomUniform(-0.05, 0.05) if kernel_initializer is None else kernel_initializer
        )
        self.bias_initializer = tf.keras.initializers.Zeros() if bias_initializer is None else bias_initializer
        self.is_channel_first = is_channel_first
        self.apply_conjugate = apply_conjugate
        self.pad_to_power_2 = pad_to_power_2
        self.method = method

        if method == MultiplicationType.MATRIX_PRODUCT:
            self.multiply = self._matrix_product
        elif method == MultiplicationType.ELEMENT_WISE:
            self.multiply = self._elementwise_product
        else:
            raise ValueError(f'Entered method: {self.method.name} unkown. Use "MATRIX_PRODUCT" or "ELEMENT_WISE".')

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer depending on the `input_shape` (output shape of the previous layer).

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor to this layer.
        """
        super(FourierConvolution2D, self).build(input_shape)
        if self.is_channel_first:
            self.batch_size, self.inp_filter, self.inp_height, self.inp_width = input_shape
        else:
            self.batch_size, self.inp_height, self.inp_width, self.inp_filter = input_shape

        if self.kernels is None:
            self.kernels = (self.inp_height, self.inp_width)

        # weights are independent from batch size [out_filter,inp_filter,kernel,kernel]. I leave the two kernels last, since I then can easily calculate the 2d FFT at once!
        self.kernel = self.add_weight(
            name="kernel",
            shape=[self.filters, self.inp_filter, self.kernels[0], self.kernels[1]],
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.padding = self._get_image_padding()
        self.paddedImageShape = (
            self.batch_size,
            self.inp_filter,
            self.inp_height + 2 * self.padding,
            self.inp_width + 2 * self.padding,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias", shape=[self.filters, 1, 1], initializer=self.bias_initializer, trainable=True
            )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the `FourierConvolution2D` layer.

        Args:
            inputs (tf.Tensor): Tensor of data in spatial domain of shape `(b,h,w,c)` or `(b,c,h,w)` depending on
                `is_channel_first`

        Raises:
            ValueError: If Layer has not been built.

        Returns:
            Tensor in spatial domain of same shape as input.
        """
        if not self.built:
            raise ValueError("This model has not yet been built.")

        # FFT2D is calculated over last two dimensions!
        if not self.is_channel_first:
            inputs = tf.einsum("bhwc->bchw", inputs)

        # Optionally pad to power of 2, to speed up FFT
        if self.pad_to_power_2:
            image_shape = self._fill_image_shape_power_2()

        # Compute DFFTs for both inputs and kernel weights
        inputs_f_domain = tf.signal.rfft2d(
            inputs, fft_length=[self.paddedImageShape[-2], self.paddedImageShape[-1]]
        )  # [batch,height,width,channel]
        kernels_f_domain = tf.signal.rfft2d(
            self.kernel, fft_length=[self.paddedImageShape[-2], self.paddedImageShape[-1]]
        )
        if self.apply_conjugate:
            kernels_f_domain = tf.math.conj(kernels_f_domain)  # to be equvivalent to the cross correlation

        outputs_f_domain = self.multiply(inputs_f_domain, kernels_f_domain)

        # Inverse rDFFT
        output = tf.signal.irfft2d(outputs_f_domain, fft_length=[self.paddedImageShape[-2], self.paddedImageShape[-1]])
        output = tf.roll(
            output, shift=[2 * self.padding, 2 * self.padding], axis=[-2, -1]
        )  # shift the samples to obtain linear conv from circular conv
        output = tf.slice(
            output,
            begin=[0, 0, self.padding, self.padding],
            size=[self.batch_size, self.filters, self.inp_height, self.inp_width],
        )

        # Optionally add bias
        if self.use_bias:
            output += self.bias

        # reverse the channel configuration to its initial config
        if not self.is_channel_first:
            output = tf.einsum("bchw->bhwc", output)

    def _matrix_product(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Calculates the elementwise product for all batches and filters, by reshaping and taking the matrix product.

        Info:
            Is much faster, but requires more memory!
        """
        a = tf.einsum("bchw->bhwc", a)
        a = tf.expand_dims(a, -2)  # [b,w,h,1,c]

        b = tf.einsum("oihw->hwio", b)

        # Matrix Multiplication
        c = a @ b
        c = tf.squeeze(c, axis=-2)
        return tf.einsum("bhwc->bchw", c)

    def _elementwise_product(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Calculates the element-wise product multiple times taking advantage of array broadcasting.

        Info:
            Is slower as the matrix multiplication, but requires less memory!
        """
        a = tf.einsum("bchw->bhwc", a)
        a = tf.expand_dims(a, -1)  # [b,w,h,c,1]
        b = tf.einsum("oihw->hwio", b)  # [k, k, in, out]

        c = a * b
        c = tf.einsum("bhwxc->bchwx", c)

        return tf.math.reduce_sum(c, axis=-1)

    def _get_image_padding(self) -> int:
        """Gets the amount of padding required to have the same spatial width before and after applying the convolution."""
        return int((self.kernels[0] - 1) / 2)

    def _fill_image_shape_power_2(self) -> Tuple[int, int, int, int]:
        """Pads the shape of the image to be a power of two. FFT is faster for such shapes.

        Returns:
            Tuple[int, int, int, int]: Image shape with padded dimensions.
        """
        width = self.paddedImageShape[-1]
        log2 = np.log2(width)
        new_power = int(np.ceil(log2))
        return (self.paddedImageShape[0], self.paddedImageShape[1], 2**new_power, 2**new_power)

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(FourierConvolution2D, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernels": self.kernels,
                "use_bias": self.use_bias,
                "kernel_initializer": self.kernel_initializer,
                "bias_initializer": self.bias_initializer,
                "is_channel_first": self.is_channel_first,
                "apply_conjugate": self.apply_conjugate,
                "pad_to_power_2": self.pad_to_power_2,
                "method": self.method,
            }
        )
        return config


class FourierFilter2D(tf.keras.layers.Layer):
    """Complex-valued learnable filter in frequency domain. Expects input data to be in the fourier domain.

    To transform an image into the frequency domain you can use `DeepSaki.layers.FFT2D`.

    **Example:**
    ```python hl_lines="5"
    import DeepSaki as dsk
    # pseudo code to load data
    image_dataset = load_data(data_path)
    x = dsk.layers.FFT2D()(image_dataset)
    x = dsk.layers.FourierFilter2D(filters=64)(x)
    x = dsk.layers.iFFT2D()(x)

    ```
    """

    def __init__(
        self,
        filters: int,
        use_bias: bool = True,
        filter_initializer: Optional[tf.keras.initializers.Initializer] = None,
        bias_initializer: Optional[tf.keras.initializers.Initializer] = None,
        is_channel_first: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the `FourierFilter2D` object.

        Args:
            filters (int): Number of independent filters
            use_bias (bool, optional): Whether or not to us bias weights. Defaults to `True`.
            filter_initializer (tf.keras.initializers.Initializer, optional): Initializer to initialize the wheights of
                the filter layer. Defaults to `None`.
            bias_initializer (tf.keras.initializers.Initializer, optional): Initializer to initialize the wheights of
                the bias weights. Defaults to `None`.
            is_channel_first (bool, optional): Set true if input shape is (b,c,h,w) and false if input shape is
                (b,h,w,c). Defaults to `False`.
            kwargs (Any): Additional key word arguments passed to the base class.
        """
        super(FourierFilter2D, self).__init__(**kwargs)
        self.filters = filters
        self.use_bias = use_bias

        filter_initializer = (
            tf.keras.initializers.RandomUniform(-0.05, 0.05) if filter_initializer is None else filter_initializer
        )
        bias_initializer = tf.keras.initializers.Zeros() if bias_initializer is None else bias_initializer
        self.filter_initializer = make_initializer_complex(filter_initializer)
        self.bias_initializer = make_initializer_complex(bias_initializer)
        self.is_channel_first = is_channel_first

        self.fourier_filter = None  # shape: batch, height, width, input_filters, output_filters
        self.fourier_bias = None
        self.out_shape = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer depending on the `input_shape` (output shape of the previous layer).

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor to this layer.
        """
        super(FourierFilter2D, self).build(input_shape)
        if self.is_channel_first:
            batch_size, inp_filter, inp_height, inp_width = input_shape
        else:
            batch_size, inp_height, inp_width, inp_filter = input_shape

        # weights are independent from batch size. Filter dimensions differ from convolution, since FFT2D is calculated over last 2 dimensions
        self.fourier_filter = self.add_weight(
            name="filter",
            shape=[inp_filter, inp_height, inp_width, self.filters],
            initializer=self.filter_initializer,
            trainable=True,
            dtype=tf.dtypes.complex64,
        )

        if (
            self.use_bias
        ):  # shape: [filter,1,1] so it can be broadcasted when adding to the output, since FFT asumes channel first!
            self.fourier_bias = self.add_weight(
                name="bias",
                shape=[self.filters, 1, 1],
                initializer=self.bias_initializer,
                trainable=True,
                dtype=tf.dtypes.complex64,
            )

        # Output shape: batch_size, self.filters, inp_height, inp_width. Filters is zero, since concatenated later
        self.out_shape = (batch_size, 0, inp_height, inp_width)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """I take advantage of broadcasting to calculate the batches: https://numpy.org/doc/stable/user/basics.broadcasting.html"""
        if not self.built:
            raise ValueError("This model has not yet been built.")

        if not self.is_channel_first:  # FFT2D is calculated over last two dimensions!
            inputs = tf.einsum("bhwc->bchw", inputs)

        output = np.ndarray(shape=self.out_shape)
        for current_filter in range(self.filters):
            # inputs:(batch, inp_filter, height, width ), fourier_filter:(...,inp_filter,height, width, out_filter)
            output = tf.concat(
                [
                    output,
                    tf.reduce_sum(
                        inputs * self.fourier_filter[:, :, :, current_filter],
                        axis=-3,  # sum over all applied filters
                        keepdims=True,
                    ),
                ],
                axis=-3,  # is the new filter count, since channel first
            )

        if self.use_bias:
            output += self.fourier_bias

        if not self.is_channel_first:  # reverse the channel configuration to its initial config
            output = tf.einsum("bchw->bhwc", output)

        return output

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(FourierFilter2D, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "use_bias": self.use_bias,
                "kernel_initializer": self.filter_initializer,
                "bias_initializer": self.bias_initializer,
                "is_channel_first": self.is_channel_first,
            }
        )
        return config


class FFT2D(tf.keras.layers.Layer):
    """Calculates the 2D descrete fourier transform over the 2 innermost channels.

    For a 4D input of shape (batch, height, width, channel), the 2DFFT would be calculated over (height, width)
    """

    def __init__(
        self,
        is_channel_first: bool = False,
        apply_real_fft: bool = False,
        shift_fft: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the `FFT2D` layer.

        Args:
            is_channel_first (bool, optional): Set true if input shape is (b,c,h,w) and false if input shape is
                (b,h,w,c). Defaults to `False`.
            apply_real_fft (bool, optional): If True, rfft2D is applied, which assumes real valued inputs and halves the
                width of the output. If False, fft2D is applied, which assumes complex input. Defaults to False.
            shift_fft (bool, optional): If true, low frequency componentes are centered. Defaults to True.
            kwargs (Any): keyword arguments passed to the parent class tf.keras.layers.Layer.
        """
        super(FFT2D, self).__init__(**kwargs)
        self.is_channel_first = is_channel_first
        self.apply_real_fft = apply_real_fft
        self.shift_fft = shift_fft
        self.policy_compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.cast(inputs, tf.float32)  # mixed precision not supported with float16
        if not self.is_channel_first:
            inputs = tf.einsum("bhwc->bchw", inputs)

        if self.apply_real_fft:
            x = tf.signal.rfft2d(inputs)
            if self.shift_fft:
                x = tf.signal.fftshift(x, axes=[-2])
        else:
            imag = tf.zeros_like(inputs)
            inputs = tf.complex(inputs, imag)  # fft2d requires complex inputs -> create complex with 0 imaginary
            x = tf.signal.fft2d(inputs)
            if self.shift_fft:
                x = tf.signal.fftshift(x)

        if not self.is_channel_first:  # reverse the channel configuration to its initial config
            x = tf.einsum("bchw->bhwc", x)
        return tf.cast(tf.math.real(x), self.policy_compute_dtype)

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(FFT2D, self).get_config()
        config.update(
            {
                "is_channel_first": self.is_channel_first,
                "apply_real_fft": self.apply_real_fft,
                "shift_fft": self.shift_fft,
                "policy_compute_dtype": self.policy_compute_dtype,
            }
        )
        return config


class FFT3D(tf.keras.layers.Layer):
    """Calculates the 3D descrete fourier transform over the 3 innermost channels.

    For a 4D input like a batch of images of shape (batch, height, width, channel), the 3DFFT would be calculated over
    (height, width).
    For a 5D input, like a batch of videos of shape (batch, frame, height, width, channel), the 3DFFT would be
    calculated over (frame, height, width).

    """

    def __init__(self, apply_real_fft: bool = False, shift_fft: bool = True, **kwargs: Any) -> None:
        """Initializes the `FFT3D` layer.

        Args:
            apply_real_fft (bool, optional): If True, rfft3D is applied, which assumes real valued inputs and halves the
                width of the output. If False, fft3D is applied, which assumes complex input. Defaults to False.
            shift_fft (bool, optional): If true, low frequency componentes are centered. Defaults to True.
            kwargs (Any): keyword arguments passed to the parent class tf.keras.layers.Layer.
        """
        super(FFT3D, self).__init__(**kwargs)
        self.apply_real_fft = apply_real_fft
        self.shift_fft = shift_fft
        self.policy_compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.cast(inputs, tf.float32)  # mixed precision not supported with float16

        if self.apply_real_fft:
            x = tf.signal.rfft3d(inputs)
            if self.shift_fft:
                x = tf.signal.fftshift(x, axes=[-2])
        else:
            imag = tf.zeros_like(inputs)
            inputs = tf.complex(inputs, imag)  # fft3d requires complex inputs -> create complex with 0 imaginary
            x = tf.signal.fft3d(inputs)
            if self.shift_fft:
                x = tf.signal.fftshift(x)

        return tf.cast(tf.math.real(x), self.policy_compute_dtype)

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(FFT3D, self).get_config()
        config.update(
            {
                "apply_real_fft": self.apply_real_fft,
                "shift_fft": self.shift_fft,
                "policy_compute_dtype": self.policy_compute_dtype,
            }
        )
        return config


class iFFT2D(tf.keras.layers.Layer):
    """Calculates the 2D inverse FFT."""

    def __init__(
        self,
        is_channel_first: bool = False,
        apply_real_fft: bool = False,
        shift_fft: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the `iFFT2D` layer.

        Args:
            is_channel_first (bool, optional): Set true if input shape is (b,c,h,w) and false if input shape is
                (b,h,w,c). Defaults to `False`.
            apply_real_fft (bool, optional): If True, rfft2D is applied, which assumes real valued inputs and halves the
                width of the output. If False, fft2D is applied, which assumes complex input. Defaults to False.
            shift_fft (bool, optional): If true, low frequency componentes are centered. Defaults to True.
            kwargs (Any): keyword arguments passed to the parent class tf.keras.layers.Layer.
        """
        super(iFFT2D, self).__init__(**kwargs)
        self.is_channel_first = is_channel_first
        self.apply_real_fft = apply_real_fft
        self.shift_fft = shift_fft

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if not self.is_channel_first:
            inputs = tf.einsum("bhwc->bchw", inputs)
        x = inputs

        if self.apply_real_fft:
            if self.shift_fft:
                x = tf.signal.ifftshift(x, axes=[-2])
            x = tf.signal.irfft2d(x)
        else:
            if self.shift_fft:
                x = tf.signal.ifftshift(x)
            x = tf.signal.ifft2d(x)

        x = tf.math.real(x)

        if not self.is_channel_first:  # reverse the channel configuration to its initial config
            x = tf.einsum("bchw->bhwc", x)
        return x

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(iFFT2D, self).get_config()
        config.update(
            {
                "is_channel_first": self.is_channel_first,
                "apply_real_fft": self.apply_real_fft,
                "shift_fft": self.shift_fft,
            }
        )
        return config


class iFFT3D(tf.keras.layers.Layer):
    """Calculates the 3D inverse FFT."""

    def __init__(
        self,
        apply_real_fft: bool = False,
        shift_fft: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the `iFFT3D` layer.

        Args:
            apply_real_fft (bool, optional): If True, rfft3D is applied, which assumes real valued inputs and halves the
                width of the output. If False, fft3D is applied, which assumes complex input. Defaults to False.
            shift_fft (bool, optional): If true, low frequency componentes are centered. Defaults to True.
            kwargs (Any): keyword arguments passed to the parent class tf.keras.layers.Layer.
        """
        super(iFFT3D, self).__init__(**kwargs)
        self.apply_real_fft = apply_real_fft
        self.shift_fft = shift_fft

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs

        if self.apply_real_fft:
            if self.shift_fft:
                x = tf.signal.ifftshift(x, axes=[-2])
            x = tf.signal.irfft3d(x)
        else:
            if self.shift_fft:
                x = tf.signal.ifftshift(x)
            x = tf.signal.ifft3d(x)

        return tf.math.real(x)

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(iFFT3D, self).get_config()
        config.update({"apply_real_fft": self.apply_real_fft, "shift_fft": self.shift_fft})
        return config
