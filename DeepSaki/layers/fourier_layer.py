
from typing import Any
from typing import Callable
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Tuple

import numpy as np
import tensorflow as tf

from DeepSaki.initializers.initializer_helper import make_initializer_complex

from DeepSaki.types.layers_enums import MultiplicationType, FrequencyFilter


# Base class below has no init. so if subclass calls super().__init__ it takes the one of tf.keras.Layer.
class FourierLayer(tf.keras.layers.Layer):
    """Base Class for Fourier Layers."""

    def _change_to_channel_first(self, input_tensor: tf.Tensor) -> tf.Tensor:
        rank = tf.rank(input_tensor)
        if rank == 4:
            return tf.einsum("bhwc->bchw", input_tensor)

        if rank == 5:
            return tf.einsum("bfhwc->bcfhw", input_tensor)

        raise ValueError(f"Only supported for rank 4 and 5 tensors but got {rank=}")

    def _change_to_channel_last(self, input_tensor: tf.Tensor) -> tf.Tensor:
        rank = tf.rank(input_tensor)
        if rank == 4:
            return tf.einsum("bchw->bhwc", input_tensor)

        if rank == 5:
            return tf.einsum("bcfhw->bfhwc", input_tensor)

        raise ValueError(f"Only supported for rank 4 and 5 tensors but got {rank=}")

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
        return tf.einsum("bhwo->bohw", c)

    def _elementwise_product(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Calculates the element-wise product multiple times taking advantage of array broadcasting.

        Info:
            Is slower as the matrix multiplication, but requires less memory!
        """
        a = tf.einsum("bchw->bhwc", a)  # [b,c,h,w] -> [b,h,w,c]
        a = tf.expand_dims(a, -1)  # [b,h,w,c] -> [b,h,w,c,1]
        # in & out are the number of input and output filters of the conv layer. in = channels, out=filters
        b = tf.einsum("oihw->hwio", b)  # [in ,out h, w] -> [h, w, in, out]

        c = a * b
        c = tf.einsum("bhwco->bohwc", c)

        return tf.math.reduce_sum(c, axis=-1)

    def _get_multiplication_function(
        self, multiplication_type: MultiplicationType
    ) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        """Returns the corresponding elementwise multiplication function for a given type."""
        valid_multiplication_types = {
            MultiplicationType.MATRIX_PRODUCT: self._matrix_product,
            MultiplicationType.ELEMENT_WISE: self._elementwise_product,
        }

        if multiplication_type not in valid_multiplication_types:
            raise ValueError(
                f"Entered multiplication_type: {self.multiplication_type.name} unkown. Valid options are: {valid_multiplication_types.keys()}"
            )
        return valid_multiplication_types.get(multiplication_type)


class FourierConvolution2D(FourierLayer):
    """Performs a convolution by transforming into fourier domain. Layer input is asumed to be in spatial domain."""

    def __init__(
        self,
        filters: int = 3,
        kernels: Optional[Tuple[int, int]] = None,
        use_bias: bool = True,
        is_channel_first: bool = False,
        apply_conjugate: bool = False,
        pad_to_power_2: bool = True,
        multiplication_type: MultiplicationType = MultiplicationType.ELEMENT_WISE,
        kernel_initializer: Optional[tf.keras.initializers.Initializer] = None,
        bias_initializer: Optional[tf.keras.initializers.Initializer] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the `FourierConvolution2D` object.

        Args:
            filters (int, optional): Number of individual filters. Defaults to 3.
            kernels (Optional[Tuple[int, int]], optional): Kernel of the spatial convolution. Expected input
                `[height,width]`. If `None`, kernel size is set to the input height and width. Defaults to `None`.
            use_bias (bool, optional): Whether or not to us bias weights. Defaults to `True`.
            is_channel_first (bool, optional): Set true if input shape is (b,c,h,w) and false if input shape is
                (b,h,w,c). Defaults to `False`.
            apply_conjugate (bool, optional): If true, the kernels are conjugated. If so, a multiplication in the
                frequency domain corresponds to a cross correlation in the spatial domain, which is actually what a
                convolution layer is doing. Defaults to `False`.
            pad_to_power_2 (bool, optional): If true, input tensor is padded. FFT algorithm runs faster for lengths of
                power of two. Defaults to `True`.
            multiplication_type (MultiplicationType, optional): Type of algo used for the element wise multiplication
                and reduction of the input and the convolution kernel. [`MultiplicationType.ELEMENT_WISE` |
                `MultiplicationType.MATRIX_PRODUCT`]. MATRIX_PRODUCT is faster, but requires more memory. ELEMENT_WISE
                is slower but requires less memory. Defaults to `MultiplicationType.ELEMENT_WISE`.
            kernel_initializer (tf.keras.initializers.Initializer, optional): Initializer to initialize the kernels of
                the convolution layer. Defaults to `None`.
            bias_initializer (tf.keras.initializers.Initializer, optional): Initializer to initialize the bias weights
                of the convolution layer. Defaults to `None`.
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
        self.multiplication_type = multiplication_type

        self.input_spec = tf.keras.layers.InputSpec(ndim=4)

        self.multiply = self._get_multiplication_function(multiplication_type)

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
        self.padding = self._get_image_padding(self.kernels)
        self.paddedImageShape = (
            self.batch_size,
            self.inp_filter,
            self.inp_height + 2 * self.padding[0],
            self.inp_width + 2 * self.padding[1],
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

        Returns:
            Tensor in spatial domain of same shape as input.
        """
        # FFT2D is calculated over last two dimensions!
        if not self.is_channel_first:
            inputs = self._change_to_channel_first(inputs)

        # Optionally pad to power of 2, to speed up FFT
        if self.pad_to_power_2:
            self.paddedImageShape = self._fill_image_shape_power_2(self.paddedImageShape)

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
        # shift the samples to obtain linear conv from circular conv
        output = tf.roll(output, shift=[2 * self.padding[0], 2 * self.padding[1]], axis=[-2, -1])

        # obtain initial shape by removing padding
        output = tf.slice(
            output,
            begin=[0, 0, self.padding[0], self.padding[1]],
            size=[self.batch_size, self.filters, self.inp_height, self.inp_width],
        )

        # Optionally add bias
        if self.use_bias:
            output += self.bias

        # reverse the channel configuration to its initial config
        if not self.is_channel_first:
            output = self._change_to_channel_last(output)

        return output

    def _get_image_padding(self, kernels: tf.Tensor) -> Tuple[int, int]:
        """Gets the amount of padding required to have the same spatial width before and after applying the convolution."""
        return (int((kernels[0] - 1) / 2), int((kernels[1] - 1) / 2))

    def _fill_image_shape_power_2(self, tensor_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Pads the shape of the image to be a power of two. FFT is faster for such shapes.

        Args:
            tensor_shape (Tuple[int,int]): Shape of the tensor (b, c, h, w)

        Returns:
            Tuple[int, int, int, int]: Image shape with padded dimensions.
        """
        new_power_width = int(np.ceil(np.log2(tensor_shape[3])))
        new_power_height = int(np.ceil(np.log2(tensor_shape[2])))
        return (tensor_shape[0], tensor_shape[1], 2**new_power_height, 2**new_power_width)

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
                "multiplication_type": self.multiplication_type,
            }
        )
        return config


class FourierFilter2D(FourierLayer):
    """Complex-valued learnable filter in frequency domain. Expects input data to be in the fourier domain.

    The filter has the same size as the input and can hence act as any type of filter: low-pass, high-pass, band-pass,
    band-stop or any combination.

    To transform an image into the frequency domain you can use `DeepSaki.layers.FFT2D`.

    **Example:**
    ```python hl_lines="5"
    import DeepSaki as ds
    # pseudo code to load data
    image_dataset = load_data(data_path)
    x = ds.layers.FFT2D()(image_dataset)
    x = ds.layers.FourierFilter2D(filters=32)(x)
    x = ds.layers.FourierFilter2D(filters=64)(x)
    x = ds.layers.FourierFilter2D(filters=128)(x)
    x = ds.layers.iFFT2D()(x)

    ```
    """

    def __init__(
        self,
        filters: int = 3,
        use_bias: bool = True,
        is_channel_first: bool = False,
        multiplication_type: MultiplicationType = MultiplicationType.ELEMENT_WISE,
        filter_initializer: Optional[tf.keras.initializers.Initializer] = None,
        bias_initializer: Optional[tf.keras.initializers.Initializer] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the `FourierFilter2D` object.

        Args:
            filters (int, optional): Number of independent filters. Defaults to 3.
            use_bias (bool, optional): Whether or not to us bias weights. Defaults to `True`.
            is_channel_first (bool, optional): Set true if input shape is (b,c,h,w) and false if input shape is
                (b,h,w,c). Defaults to `False`.
            multiplication_type (MultiplicationType, optional): Type of algo used for the element wise multiplication
                and reduction of the input and the filter weights. [`MultiplicationType.ELEMENT_WISE` |
                `MultiplicationType.MATRIX_PRODUCT`]. MATRIX_PRODUCT is faster, but requires more memory. ELEMENT_WISE
                is slower but requires less memory. Defaults to `MultiplicationType.ELEMENT_WISE`.
            filter_initializer (tf.keras.initializers.Initializer, optional): Initializer to initialize the wheights of
                the filter layer. Defaults to `None`.
            bias_initializer (tf.keras.initializers.Initializer, optional): Initializer to initialize the wheights of
                the bias weights. Defaults to `None`.
            kwargs (Any): Additional key word arguments passed to the base class.
        """
        super(FourierFilter2D, self).__init__(**kwargs)
        self.filters = filters
        self.use_bias = use_bias
        self.is_channel_first = is_channel_first
        self.multiplication_type = multiplication_type

        filter_initializer = (
            tf.keras.initializers.RandomUniform(-0.05, 0.05) if filter_initializer is None else filter_initializer
        )
        bias_initializer = tf.keras.initializers.Zeros() if bias_initializer is None else bias_initializer
        self.filter_initializer = make_initializer_complex(filter_initializer)
        self.bias_initializer = make_initializer_complex(bias_initializer)
        self.multiply = self._get_multiplication_function(multiplication_type)

        self.input_spec = tf.keras.layers.InputSpec(ndim=4, dtype=tf.complex64)

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer depending on the `input_shape` (output shape of the previous layer).

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor to this layer.
        """
        super(FourierFilter2D, self).build(input_shape)
        if self.is_channel_first:
            _, channel, height, width = input_shape
        else:
            _, height, width, channel = input_shape

        self.fourier_filter = self.add_weight(
            name="filter",
            shape=[self.filters, channel, height, width],
            initializer=self.filter_initializer,
            trainable=True,
            dtype=tf.dtypes.complex64,
        )

        if self.use_bias:
            self.fourier_bias = self.add_weight(
                name="bias",
                shape=[self.filters, 1, 1],
                initializer=self.bias_initializer,
                trainable=True,
                dtype=tf.dtypes.complex64,
            )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the `FourierFilter2D` layer.

        Args:
            inputs (tf.Tensor): Complex valued tensor of shape (b,h,w,c) if `is_channel_first=False` or Tensor of shape
                (b,c,h,w) if `is_channel_first=True`.

        Returns:
            Complex valued tensor of shape (b,h,w,`filters`) if `is_channel_first=False` or Tensor of shape
                (b,`filters`,h,w) if `is_channel_first=True`.
        """
        if not self.is_channel_first:  # FFT2D is calculated over last two dimensions!
            inputs = self._change_to_channel_first(inputs)

        output = self.multiply(inputs, self.fourier_filter)

        if self.use_bias:
            output += self.fourier_bias

        if not self.is_channel_first:  # reverse the channel configuration to its initial config
            output = self._change_to_channel_last(output)

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
                "filter_initializer": self.filter_initializer,
                "bias_initializer": self.bias_initializer,
                "is_channel_first": self.is_channel_first,
                "multiplication_type": self.multiplication_type,
            }
        )
        return config


class FFT2D(FourierLayer):
    """Calculates the 2D descrete fourier transform over the 2 last dimensions.

    For a 4D input of shape (batch, channel, height, width), the 2DFFT would be calculated over (height, width).

    Know what to expect:
        | input shape | input dtype                 |   apply_real_fft   |  output dtype           | output shape |
        |-------------|-----------------------------|:------------------:|------------------------:|-------------:|
        | (1,8,8,3)   | real                        |  True              | complex                 |  (1,8,5,3)   |
        | (1,8,8,3)   | real                        |  false             | n.a.                    |     n.a.     |
        | (1,8,8,3)   | complex                     |  True(Value Error) | n.a.                    |     n.a.     |
        | (1,8,8,3)   | complex                     |  False             | complex                 |  (1,8,8,3)   |

    Limitations:
        Height and width are expected to be equal for now.
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
                width of the output(width/2+1 because center frequency is kept). If False, fft2D is applied, which
                assumes complex input. Defaults to False.
            shift_fft (bool, optional): If true, low frequency components are centered. Defaults to True.
            kwargs (Any): keyword arguments passed to the parent class tf.keras.layers.Layer.
        """
        super(FFT2D, self).__init__(**kwargs)
        self.is_channel_first = is_channel_first
        self.apply_real_fft = apply_real_fft
        self.shift_fft = shift_fft
        self.policy_compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype

        dtype = tf.float32 if self.apply_real_fft else tf.complex64
        self.input_spec = tf.keras.layers.InputSpec(ndim=4, dtype=dtype)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        r"""Calls the `FFT2D` layer.

        Args:
            inputs (tf.Tensor): Tensor of shape (b,h,w,c) if `is_channel_first=False` or Tensor of shape (b,c,h,w) if
                `is_channel_first=True`. `h` and `w` should be equal for now.

        Returns:
            Tensor of shape (b,h,Wo,c) if `is_channel_first=False` or  Tensor of shape (b,c,h,Wo) if
                `is_channel_first=True`. If `apply_real_fft=True` $Wo=\frac{w}{2}+1$.
        """
        if not self.is_channel_first:
            inputs = self._change_to_channel_first(inputs)

        if self.apply_real_fft:
            x = tf.signal.rfft2d(inputs)
            if self.shift_fft:
                x = tf.signal.fftshift(x, axes=[-2])
        else:
            if inputs.dtype not in [tf.complex64, tf.complex128]:
                imag = tf.zeros_like(inputs)
                inputs = tf.complex(inputs, imag)  # fft2d requires complex inputs -> create complex with 0 imaginary
            x = tf.signal.fft2d(inputs)
            if self.shift_fft:
                x = tf.signal.fftshift(x)

        if not self.is_channel_first:  # reverse the channel configuration to its initial config
            x = self._change_to_channel_last(x)

        return x

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


class FFT3D(FourierLayer):
    """Calculates the 3D descrete fourier transform over the 3 last dimensions.

    For a 4D input like a batch of images of shape (batch, channel, height, width), the 3DFFT would be calculated over
    (channel, height, width).
    For a 5D input, like a batch of videos of shape (batch, channel, frame, height, width), the 3DFFT would be
    calculated over (frame, height, width).
    """

    def __init__(
        self,
        is_channel_first: bool = False,
        apply_real_fft: bool = False,
        shift_fft: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the `FFT3D` layer.

        Args:
            is_channel_first (bool, optional): Set true if input shape is (b,c,h,w) or (b,c,f,w,h) and false if input
                shape is (b,h,w,c) or (b,f,w,h,c). Defaults to `False`.
            apply_real_fft (bool, optional): If True, rfft3D is applied, which assumes real valued inputs and halves the
                width of the output. If False, fft3D is applied, which assumes complex input. Defaults to False.
            shift_fft (bool, optional): If true, low frequency components are centered. Defaults to True.
            kwargs (Any): keyword arguments passed to the parent class tf.keras.layers.Layer.
        """
        super(FFT3D, self).__init__(**kwargs)
        self.is_channel_first = is_channel_first
        self.apply_real_fft = apply_real_fft
        self.shift_fft = shift_fft
        self.policy_compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
        dtype = tf.float32 if self.apply_real_fft else tf.complex64
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=4, max_ndim=5, dtype=dtype)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        r"""Calls the `FFT3D` layer.

        Args:
            inputs (tf.Tensor): Tensor of shape (b,h,w,c) or (b,f,w,h,c) if `is_channel_first=False` or  Tensor of
                shape (b,c,h,w) or (b,c,f,w,h) if `is_channel_first=True`. `h` and `w` should be equal for now.

        Returns:
            Tensor of shape (b,h,Wo,c) or (b,f,h,Wo,c) if `is_channel_first=False` or Tensor of shape (b,c,h,Wo) or
                (b,f,c,h,Wo) if `is_channel_first=True`. If `apply_real_fft=True` $Wo=\frac{w}{2}+1$.
        """
        if not self.is_channel_first:
            inputs = self._change_to_channel_first(inputs)

        if self.apply_real_fft:
            x = tf.signal.rfft3d(inputs)
            if self.shift_fft:
                x = tf.signal.fftshift(x, axes=[-2])
        else:
            if inputs.dtype not in [tf.complex64, tf.complex128]:
                imag = tf.zeros_like(inputs)
                inputs = tf.complex(inputs, imag)  # fft3d requires complex inputs -> create complex with 0 imaginary
            x = tf.signal.fft3d(inputs)
            if self.shift_fft:
                x = tf.signal.fftshift(x)

        if not self.is_channel_first:
            x = self._change_to_channel_last(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(FFT3D, self).get_config()
        config.update(
            {
                "is_channel_first": self.is_channel_first,
                "apply_real_fft": self.apply_real_fft,
                "shift_fft": self.shift_fft,
                "policy_compute_dtype": self.policy_compute_dtype,
            }
        )
        return config


class iFFT2D(FourierLayer):
    """Calculates the 2D inverse FFT.

    Know what to expect:
        |   apply_real_fft   |input generated by          | input shape | input dtype  |  output dtype   | output shape |
        |:------------------:|----------------------------|-------------|--------------|----------------:|-------------:|
        |  True              | FFT2D(apply_real_fft=True) | (1,8,8,3)   | complex      | real            |  (1,8,5,3)   |
        |  false             | FFT2D(apply_real_fft=False)| (1,8,8,3)   | complex      | complex         |  (1,8,8,3)   |

    **Example:**
    ```python hl_lines="5"
    import DeepSaki as ds
    import tensorflow as tf

    real_data = tf.random.normal(shape=(8,64,64,3))
    complex_data = tf.complex(real_data,real_data)

    # Real FFT with real valued data
    x = ds.layers.FFT2D(apply_real_fft = True)(real_data) #<tf.Tensor: shape=(8, 64, 33, 3), dtype=complex64>
    x = ds.layers.iFFT2D(apply_real_fft = True)(x) #<tf.Tensor: shape=(8, 64, 64, 3), dtype=float32>

    # Standard FFT with complex valued data
    x = ds.layers.FFT2D(apply_real_fft = False)(complex_data) #<tf.Tensor: shape=(8, 64, 64, 3), dtype=complex64>
    x = ds.layers.iFFT2D(apply_real_fft = False)(x) #<tf.Tensor: shape=(8, 64, 64, 3), dtype=complex64>

    # Standard FFT with real valued data - FFT2D will create a pseude complex tensor tf.complex(real, tf.zeros_like(real))
    x = ds.layers.FFT2D(apply_real_fft = False)(real_data) #<tf.Tensor: shape=(8, 64, 64, 3), dtype=complex64>
    x = ds.layers.iFFT2D(apply_real_fft = False)(x) #<tf.Tensor: shape=(8, 64, 64, 3), dtype=complex64>
    x = tf.math.real(x) # can be casted to real without issues, since imag values are all zero
    ```

    """

    def __init__(
        self,
        is_channel_first: bool = False,
        apply_real_fft: bool = False,
        shift_fft: bool = True,
        fft_length: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the `iFFT2D` layer.

        Args:
            is_channel_first (bool, optional): Set true if input shape is (b,c,h,w) and false if input shape is
                (b,h,w,c). Defaults to `False`.
            apply_real_fft (bool, optional): If True, rfft2D is applied, which assumes real valued inputs and halves the
                width of the output. If False, fft2D is applied, which assumes complex input. Defaults to False.
            shift_fft (bool, optional): If true, low frequency components are centered. Defaults to True.
            fft_length (Optional[Tuple[int,int]]): The FFT length for each dimension of the real FFT. Defaults to None.
            kwargs (Any): keyword arguments passed to the parent class tf.keras.layers.Layer.
        """
        super(iFFT2D, self).__init__(**kwargs)
        self.is_channel_first = is_channel_first
        self.apply_real_fft = apply_real_fft
        self.shift_fft = shift_fft
        self.fft_length = fft_length
        self.input_spec = tf.keras.layers.InputSpec(ndim=4, dtype=tf.complex64)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the `iFFT2D` layer.

        Args:
            inputs (tf.Tensor): Complex valued input Tensor. Shape depends on attributes `is_channel_first` and `apply_real_fft`. <br>
                `is_channel_first=False`, `apply_real_fft=False`: Expected shape of input tensor: (b,h,w,c) <br>
                `is_channel_first=True`, `apply_real_fft=False`: Expected shape of input tensor: (b,c,h,w) <br>
                `is_channel_first=False`, `apply_real_fft=True`: Expected shape of input tensor: (b,h,w=h/2+1,c) <br>
                `is_channel_first=True`, `apply_real_fft=True`: Expected shape of input tensor: (b,c,h,w=h/2+1)

        Returns:
            Tensor of shape: <br>
                `is_channel_first=False`: Expected shape of input tensor: (b,h,w=h,c) <br>
                `is_channel_first=True`: Expected shape of input tensor: (b,c,h,w=h)
        """
        if not self.is_channel_first:
            inputs = self._change_to_channel_first(inputs)
        x = inputs

        if self.apply_real_fft:
            if self.shift_fft:
                x = tf.signal.ifftshift(x, axes=[-2])
            x = tf.signal.irfft2d(x, fft_length=self.fft_length)
        else:
            if self.shift_fft:
                x = tf.signal.ifftshift(x)
            x = tf.signal.ifft2d(x)

        if not self.is_channel_first:  # reverse the channel configuration to its initial config
            x = self._change_to_channel_last(x)

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
                "fft_length": self.fft_length,
            }
        )
        return config


class iFFT3D(FourierLayer):
    """Calculates the 3D inverse FFT."""

    def __init__(
        self,
        is_channel_first: bool = False,
        apply_real_fft: bool = False,
        shift_fft: bool = True,
        fft_length: Optional[Tuple[int, int, int]] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the `iFFT3D` layer.

        Args:
            is_channel_first (bool, optional): Set true if input shape is (b,c,h,w) or (b,c,f,h,w) and false if input
                shape is (b,h,w,c) or (b,f,h,w,c). Defaults to `False`.
            apply_real_fft (bool, optional): If True, rfft3D is applied, which assumes real valued inputs and halves the
                width of the output. If False, fft3D is applied, which assumes complex input. Defaults to False.
            shift_fft (bool, optional): If true, low frequency components are centered. Defaults to True.
            fft_length (Optional[Tuple[int,int,int]]): The FFT length for each dimension of the real FFT. Defaults to None.
            kwargs (Any): keyword arguments passed to the parent class tf.keras.layers.Layer.
        """
        super(iFFT3D, self).__init__(**kwargs)
        self.is_channel_first = is_channel_first
        self.apply_real_fft = apply_real_fft
        self.shift_fft = shift_fft
        self.fft_length = fft_length
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=4, max_ndim=5, dtype=tf.complex64)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the `iFFT3D` layer.

        Args:
            inputs (tf.Tensor): Complex valued input Tensor. Shape depends on attributes `is_channel_first` and `apply_real_fft`. <br>
                `is_channel_first=False`, `apply_real_fft=False`: Expected shape of input tensor: (b,h,w,c) or (b,f,h,w,c) <br>
                `is_channel_first=True`, `apply_real_fft=False`: Expected shape of input tensor: (b,c,h,w) or (b,c,f,h,w) <br>
                `is_channel_first=False`, `apply_real_fft=True`: Expected shape of input tensor: (b,h,w=h/2+1,c) or (b,f,h,w=h/2+1,c) <br>
                `is_channel_first=True`, `apply_real_fft=True`: Expected shape of input tensor: (b,c,h,w=h/2+1) or (b,c,f,h,w=h/2+1)

        Returns:
            Tensor of shape: <br>
                `is_channel_first=False`: Expected shape of input tensor: (b,h,w=h,c) or (b,f,h,w=h,c) <br>
                `is_channel_first=True`: Expected shape of input tensor: (b,c,h,w=h) or (b,c,f,h,w=h)
        """
        if not self.is_channel_first:
            inputs = self._change_to_channel_first(inputs)
        x = inputs

        if self.apply_real_fft:
            if self.shift_fft:
                x = tf.signal.ifftshift(x, axes=[-2])
            x = tf.signal.irfft3d(x, fft_length=self.fft_length)
        else:
            if self.shift_fft:
                x = tf.signal.ifftshift(x)
            x = tf.signal.ifft3d(x)

        if not self.is_channel_first:
            x = self._change_to_channel_last(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(iFFT3D, self).get_config()
        config.update(
            {
                "is_channel_first": self.is_channel_first,
                "apply_real_fft": self.apply_real_fft,
                "shift_fft": self.shift_fft,
                "fft_length": self.fft_length,
            }
        )
        return config


class FourierPooling2D(FourierLayer):
    """Pooling in frequency domain by truncating high frequencies using a center crop operation.

    Layer input is asumed to be in frequency domain and shifted, such that the center frequency is in the center of
    the grid.

    If this is the case, the center represents the frequency of 0Hz (hence an offset). The further away from the center
    the higher the frequency component. Center cropping removes high frequency components, hence can be seen as a low
    pass filter

    """

    def __init__(
        self,
        is_channel_first: bool = False,
        input_from_rfft: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes an instance of `FourierPooling2D`.

        Args:
            is_channel_first (bool, optional): If True, input shape is assumed to be (`batch`,`channel`,`height`,`width`).
                If False, input shape is assumed to be (`batch`,`height`,`width`,`channel`). Defaults to False.
            input_from_rfft (bool, optional): If true, the input spectrum is assumed to be originated from an rFFT2D
                operation.
            kwargs (Any): Additional key word arguments passed to the base class.
        """
        super(FourierPooling2D, self).__init__(**kwargs)
        self.is_channel_first = is_channel_first
        self.input_from_rfft = input_from_rfft
        self.input_spec = tf.keras.layers.InputSpec(ndim=4, dtype=tf.complex64)

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer depending on the `input_shape` (output shape of the previous layer).

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor to this layer.
        """
        super(FourierPooling2D, self).build(input_shape)
        if self.is_channel_first:
            _, channels, height, width = input_shape
        else:
            _, height, width, channels = input_shape

        offset_height = height // 4
        offset_width = 0 if self.input_from_rfft else width // 4
        target_height = height // 2
        # + 1 is important. if real FFT width is usually odd.
        target_width = (width + 1) // 2

        if self.is_channel_first:
            self.slice_begin = [0, 0, offset_height, offset_width]
            self.slice_size = [-1, channels, target_height, target_width]
        else:
            self.slice_begin = [0, offset_height, offset_width, 0]
            self.slice_size = [-1, target_height, target_width, channels]

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the `FourierPooling2D` layer.

        Args:
            inputs (tf.Tensor): Complex valued tensor of shape (`batch`,`height`,`width`,`channel`) or
                (`batch`,`channel`,`height`,`width`). Tensor is asumed to be in frequency domain of type `tf.complex64`
                or `tf.complex128`.

        Returns:
            Pooled tensor of shape (`batch`,`channel`,`height/2`,`width/2`) or (`batch`,`height/2`,`width/2`,`channel`)
        """
        return tf.slice(inputs, begin=self.slice_begin, size=self.slice_size)

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(FourierPooling2D, self).get_config()
        config.update(
            {
                "is_channel_first": self.is_channel_first,
                "input_from_rfft": self.input_from_rfft,
            }
        )
        return config


class rFFT2DFilter(FourierLayer):
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
        self.input_spec = tf.keras.layers.InputSpec(ndim=4, dtype=tf.float32)

        self.shift_fft = self.filter_type == FrequencyFilter.LOW_PASS

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer depending on the `input_shape` (output shape of the previous layer).

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor to this layer.
        """
        super(rFFT2DFilter, self).build(input_shape)
        if self.is_channel_first:
            _, _, height, width = input_shape
        else:
            _, height, width, _ = input_shape

        self.rfft2d = FFT2D(self.is_channel_first, apply_real_fft=True, shift_fft=self.shift_fft)
        self.fourier_pooling_2d = FourierPooling2D(self.is_channel_first, input_from_rfft=True)
        self.irfft2d = iFFT2D(
            self.is_channel_first, apply_real_fft=True, shift_fft=self.shift_fft, fft_length=(height, width)
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the `rFFT2DFilter` layer.

        Args:
            inputs (tf.Tensor): Tensor of shape (`batch`,`height`,`width`,`channel`) or
                (`batch`,`channel`,`height`,`width`).

        Returns:
            Filtered tensor with shape (`batch`,`channel`,`height`,`width`).
        """
        x = self.rfft2d(inputs)
        x = self.fourier_pooling_2d(x)
        return self.irfft2d(x)

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(rFFT2DFilter, self).get_config()
        config.update({"is_channel_first": self.is_channel_first, "filter_type": self.filter_type})
        return config
