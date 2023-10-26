import pytest
import os
import inspect

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf
import numpy as np
from contextlib import nullcontext as does_not_raise

from tests.DeepSaki_test.layers_test.layers_test import DeepSakiLayerChecks,CommonLayerChecks

from DeepSaki.layers.fourier_layer import (
    FourierConvolution2D,
    FourierFilter2D,
    FFT2D,
    FFT3D,
    iFFT2D,
    iFFT3D,
    FourierPooling2D,
    rFFT2DFilter,
    MultiplicationType,
)


@pytest.fixture()
def fourier_convolution_2D():
    return FourierConvolution2D()

@pytest.fixture()
def fourier_filter_2D():
    return FourierFilter2D()

@pytest.fixture()
def fft2d():
    return FFT2D()

@pytest.fixture()
def fft3d():
    return FFT3D()

@pytest.fixture()
def ifft2d():
    return iFFT2D()

@pytest.fixture()
def ifft3d():
    return iFFT3D()

@pytest.fixture()
def fourier_pooling_2d():
    return FourierPooling2D()

@pytest.fixture()
def rfft2d_filter():
    return rFFT2DFilter()


class TestFourierConvolution2D(DeepSakiLayerChecks):
    @pytest.mark.parametrize(
        ("batch", "height", "width", "channel", "filter"),
        [
            (1, 16, 16, 3, 5),
            (8, 8, 8, 12, 24),
            (3, 15, 15, 4, 8),
        ],
    )
    def test_matrix_product(
        self,
        fourier_convolution_2D,
        batch,
        height,
        width,
        channel,
        filter,
    ):
        image_tensor = tf.ones(shape=(batch, channel, height, width))
        kernel_tensor = tf.ones(shape=(filter, channel, height, width))
        result = fourier_convolution_2D._elementwise_product(image_tensor, kernel_tensor)
        expected_tensor = channel * tf.ones(shape=(batch, filter, height, width))
        assert tf.math.reduce_all(result.numpy() == pytest.approx(expected_tensor.numpy(), 0.01))

    @pytest.mark.parametrize(
        ("batch", "height", "width", "channel", "filter"),
        [
            (1, 16, 16, 3, 5),
            (8, 8, 8, 12, 24),
            (3, 15, 15, 4, 8),
        ],
    )
    def test_elementwise_product(
        self,
        fourier_convolution_2D,
        batch,
        height,
        width,
        channel,
        filter,
    ):
        image_tensor = tf.ones(shape=(batch, channel, height, width))
        kernel_tensor = tf.ones(shape=(filter, channel, height, width))
        result = fourier_convolution_2D._elementwise_product(image_tensor, kernel_tensor)
        expected_tensor = channel * tf.ones(shape=(batch, filter, height, width))
        assert tf.math.reduce_all(result.numpy() == pytest.approx(expected_tensor.numpy(), 0.01))

    @pytest.mark.parametrize(
        ("batch", "height", "width", "channel", "filter"),
        [
            (1, 16, 16, 3, 5),
            (8, 8, 8, 12, 24),
            (3, 15, 15, 4, 8),
        ],
    )
    def test_matrix_product_same_as_element_wise(
        self,
        fourier_convolution_2D,
        batch,
        height,
        width,
        channel,
        filter,
    ):
        image_tensor = tf.ones(shape=(batch, channel, height, width))
        kernel_tensor = tf.ones(shape=(filter, channel, height, width))
        result_element_wise = fourier_convolution_2D._elementwise_product(image_tensor, kernel_tensor)
        result_matrix_product = fourier_convolution_2D._matrix_product(image_tensor, kernel_tensor)
        assert tf.math.reduce_all(result_element_wise.numpy() == pytest.approx(result_matrix_product.numpy(), 0.01))

    @pytest.mark.parametrize(
        ("kernels", "expected"),
        [
            ((3, 3), (1, 1)),
            ((5, 5), (2, 2)),
            ((7, 7), (3, 3)),
            ((9, 9), (4, 4)),
            ((5, 3), (2, 1)),
            ((3, 5), (1, 2)),
            ((7, 5), (3, 2)),
            ((5, 7), (2, 3)),
        ],
    )
    def test_get_image_padding(self,fourier_convolution_2D, kernels, expected):
        padding = fourier_convolution_2D._get_image_padding(kernels)
        assert padding == expected

    @pytest.mark.parametrize(
        ("input_shape", "expected_shape"),
        [
            ((1, 2, 8, 8), (1, 2, 8, 8)),
            ((1, 2, 8, 10), (1, 2, 8, 16)),
            ((1, 2, 17, 19), (1, 2, 32, 32)),
            ((8, 5, 50, 64), (8, 5, 64, 64)),
            ((4, 10, 77, 55), (4, 10, 128, 64)),
        ],
    )
    def test_fill_image_shape_power_2(self,fourier_convolution_2D, input_shape, expected_shape):
        output_shape = fourier_convolution_2D._fill_image_shape_power_2(input_shape)
        assert output_shape == expected_shape

    @pytest.mark.parametrize(
        ("multiplication_type", "expected_context"),
        [
            (MultiplicationType.ELEMENT_WISE, does_not_raise()),
            (MultiplicationType.MATRIX_PRODUCT, does_not_raise()),
            (None, pytest.raises(AttributeError)),
            ("Any wrong String", pytest.raises(AttributeError)),
            (1, pytest.raises(AttributeError)),
            (2, pytest.raises(AttributeError)),
        ],
    )
    def test_init_raises_error_wrong_multiplication_type(self, multiplication_type, expected_context):
        with expected_context:
            _ = FourierConvolution2D(multiplication_type=multiplication_type)

    @pytest.mark.parametrize(
        ("input_shape", "expected_context"),
        [
            (tf.TensorShape((8, 64, 64, 4)), does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3)), does_not_raise()),
            (tf.TensorShape((8, 64, 64)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 64)), pytest.raises(ValueError)),
            (tf.TensorShape((8)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5, 6)), pytest.raises(ValueError)),
        ],
    )
    def test_call_raises_error_wrong_input_dim(self,fourier_convolution_2D, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_dim(fourier_convolution_2D,input_shape,expected_context)

    @pytest.mark.parametrize("kernels", [(3, 3), (9, 9), (16, 16)])
    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize("apply_conjugate", [True, False])
    @pytest.mark.parametrize("pad_to_power_2", [True, False])
    @pytest.mark.parametrize("method", [MultiplicationType.ELEMENT_WISE, MultiplicationType.MATRIX_PRODUCT])
    @pytest.mark.parametrize(
        ("input_shape", "filters", "is_channel_first", "expected_shape"),
        [
            (tf.TensorShape((1, 16, 16, 3)), 5, False, tf.TensorShape((1, 16, 16, 5))),
            (tf.TensorShape((1, 34, 30, 16)), 9, False, tf.TensorShape((1, 34, 30, 9))),
            (tf.TensorShape((8, 16, 16, 3)), 3, False, tf.TensorShape((8, 16, 16, 3))),
            (tf.TensorShape((1, 5, 16, 16)), 5, True, tf.TensorShape((1, 5, 16, 16))),
            (tf.TensorShape((1, 16, 17, 34)), 9, True, tf.TensorShape((1, 9, 17, 34))),
            (tf.TensorShape((8, 3, 16, 16)), 3, True, tf.TensorShape((8, 3, 16, 16))),
        ],
    )
    def test_call_correct_output_shape(
        self,
        input_shape,
        filters,
        kernels,
        use_bias,
        is_channel_first,
        apply_conjugate,
        pad_to_power_2,
        method,
        expected_shape,
    ):
        layer_instance = FourierConvolution2D(
            filters, kernels, use_bias, is_channel_first, apply_conjugate, pad_to_power_2, method,
        )
        CommonLayerChecks.has_call_correct_output_shape(layer_instance, input_shape, expected_shape)


class TestFourierFilter2D(DeepSakiLayerChecks):
    @pytest.mark.xfail(reason="test not implemented")
    def test_call_raises_error_wrong_input_dim(self):
        ...

    @pytest.mark.xfail(reason="test not implemented")
    def test_call_correct_output_shape(self):
        ...

class TestFFT2D(DeepSakiLayerChecks):
    @pytest.mark.parametrize(
        ("input_shape", "expected_context"),
        [
            (tf.TensorShape((8, 64, 64, 4)), does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3)), does_not_raise()),
            (tf.TensorShape((8, 64, 64)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 64)), pytest.raises(ValueError)),
            (tf.TensorShape((8)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5, 6)), pytest.raises(ValueError)),
        ],
    )
    def test_call_raises_error_wrong_input_dim(self, fft2d, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_dim(fft2d,input_shape, expected_context)

    @pytest.mark.parametrize("shift_fft",[True,False])
    @pytest.mark.parametrize(
        ("input_shape","is_channel_first","apply_real_fft","expected_shape"),[
            (tf.TensorShape((1, 16, 16, 3)),False,False,tf.TensorShape((1, 16, 16, 3))),
            (tf.TensorShape((1, 34, 34, 16)),False,False,tf.TensorShape((1, 34, 34, 16))),
            (tf.TensorShape((8, 16, 16, 3)),False,False,tf.TensorShape((8, 16, 16, 3))),
            (tf.TensorShape((1, 5, 16, 16)),True,False,tf.TensorShape((1, 5, 16, 16))),
            (tf.TensorShape((1, 16, 34, 34)),True,False,tf.TensorShape((1, 16, 34, 34))),
            (tf.TensorShape((8, 3, 16, 16)),True,False,tf.TensorShape((8, 3, 16, 16))),
            (tf.TensorShape((1, 16, 16, 3)),False,True,tf.TensorShape((1, 16, 9, 3))),
            (tf.TensorShape((1, 34, 34, 16)),False,True,tf.TensorShape((1, 34, 18, 16))),
            (tf.TensorShape((8, 16, 16, 3)),False,True,tf.TensorShape((8, 16, 9, 3))),
            (tf.TensorShape((1, 5, 16, 16)),True,True,tf.TensorShape((1, 5, 16, 9))),
            (tf.TensorShape((1, 16, 34, 34)),True,True,tf.TensorShape((1, 16, 34, 18))),
            (tf.TensorShape((8, 3, 16, 16)),True,True,tf.TensorShape((8, 3, 16, 9))),
        ]
    )
    def test_call_correct_output_shape(self,input_shape,is_channel_first,apply_real_fft,shift_fft,expected_shape):
        layer_instance = FFT2D(is_channel_first,apply_real_fft,shift_fft)
        CommonLayerChecks.has_call_correct_output_shape(layer_instance,input_shape,expected_shape)

    @pytest.mark.parametrize(
        ("input_shape","is_channel_first","expected_context"),[
            (tf.TensorShape((1, 16, 16, 3)),False,does_not_raise()),
            (tf.TensorShape((1, 34, 30, 16)),False,pytest.raises(ValueError)),
            (tf.TensorShape((8, 16, 16, 3)),False,does_not_raise()),
            (tf.TensorShape((1, 5, 16, 16)),True,does_not_raise()),
            (tf.TensorShape((1, 16, 17, 34)),True,pytest.raises(ValueError)),
            (tf.TensorShape((8, 3, 16, 16)),True,does_not_raise()),
        ]
    )
    def test_build_raises_error_for_unsuported_shape(self, input_shape,is_channel_first, expected_context):
        layer_instance = FFT2D(is_channel_first)
        with expected_context:
            layer_instance.build(input_shape)

class TestFFT3D(DeepSakiLayerChecks):
    @pytest.mark.parametrize(
        ("input_shape", "expected_context"),
        [
            (tf.TensorShape((8, 64, 64, 4)), does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3)), does_not_raise()),
            (tf.TensorShape((8, 64, 64)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 64)), pytest.raises(ValueError)),
            (tf.TensorShape((8)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5)), does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3, 9)), does_not_raise()),
            (tf.TensorShape((8, 64, 64, 4, 5, 6)), pytest.raises(ValueError)),
        ],
    )
    def test_call_raises_error_wrong_input_dim(self, fft3d, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_dim(fft3d, input_shape, expected_context)

    @pytest.mark.parametrize("shift_fft",[True,False])
    @pytest.mark.parametrize(
        ("input_shape","is_channel_first","apply_real_fft","expected_shape"),[
            (tf.TensorShape((1, 16, 16, 3)),False,False,tf.TensorShape((1, 16, 16, 3))),
            (tf.TensorShape((8, 34, 34, 16)),False,False,tf.TensorShape((8, 34, 34, 16))),
            (tf.TensorShape((1, 5, 16, 16)),True,False,tf.TensorShape((1, 5, 16, 16))),
            (tf.TensorShape((8, 16, 34, 34)),True,False,tf.TensorShape((8, 16, 34, 34))),
            (tf.TensorShape((1, 16, 16, 3)),False,True,tf.TensorShape((1, 16, 9, 3))),
            (tf.TensorShape((8, 34, 34, 16)),False,True,tf.TensorShape((8, 34, 18, 16))),
            (tf.TensorShape((1, 5, 16, 16)),True,True,tf.TensorShape((1, 5, 16, 9))),
            (tf.TensorShape((8, 16, 34, 34)),True,True,tf.TensorShape((8, 16, 34, 18))),
            (tf.TensorShape((1, 8, 16, 16, 3)),False,False,tf.TensorShape((1, 8, 16, 16, 3))),
            (tf.TensorShape((8, 1, 34, 34, 16)),False,False,tf.TensorShape((8, 1, 34, 34, 16))),
            (tf.TensorShape((1, 5, 8, 16, 16)),True,False,tf.TensorShape((1, 5, 8, 16, 16))),
            (tf.TensorShape((8, 16, 1, 34, 34)),True,False,tf.TensorShape((8, 16, 1, 34, 34))),
            (tf.TensorShape((1, 8, 16, 16, 3)),False,True,tf.TensorShape((1, 8, 16, 9, 3))),
            (tf.TensorShape((8, 1, 34, 34, 16)),False,True,tf.TensorShape((8, 1, 34, 18, 16))),
            (tf.TensorShape((1, 5, 8, 16, 16)),True,True,tf.TensorShape((1, 5, 8,16, 9))),
            (tf.TensorShape((8, 16,1, 34, 34)),True,True,tf.TensorShape((8, 16,1, 34, 18))),
        ]
    )
    def test_call_correct_output_shape(self,input_shape,is_channel_first,apply_real_fft,shift_fft,expected_shape):
        layer_instance = FFT3D(is_channel_first,apply_real_fft,shift_fft)
        CommonLayerChecks.has_call_correct_output_shape(layer_instance,input_shape,expected_shape)

    @pytest.mark.parametrize(
        ("input_shape","expected_shape"),[
            (tf.TensorShape((1,16,16,3)), tf.TensorShape((1,3,16,16))),
            (tf.TensorShape((1,8,16,16,3)), tf.TensorShape((1,3,8,16,16))),
        ]
    )
    def test_change_to_channel_first_correct_shape(self,fft3d, input_shape, expected_shape):
        resulting_tensor = fft3d._change_to_channel_first(tf.ones(shape=input_shape))
        assert resulting_tensor.shape == expected_shape

    @pytest.mark.parametrize(
        ("input_shape","expected_context"),[
            (tf.TensorShape((1,16,16,3)), does_not_raise()),
            (tf.TensorShape((1,8,16,16,3)), does_not_raise()),
            (tf.TensorShape((1,8,16)), pytest.raises(ValueError)),
            (tf.TensorShape((1,8,16,16,3,6)), pytest.raises(ValueError)),
        ]
    )
    def test_change_to_channel_first_raises_error(self,fft3d,input_shape, expected_context):
        with expected_context:
            _ = fft3d._change_to_channel_first(tf.ones(shape=input_shape))

    @pytest.mark.parametrize(
        ("input_shape","expected_shape"),[
            (tf.TensorShape((1,3,16,16)), tf.TensorShape((1,16,16,3))),
            (tf.TensorShape((1,3,8,16,16)), tf.TensorShape((1,8,16,16,3))),
        ]
    )
    def test_change_to_channel_last_correct_shape(self, fft3d, input_shape, expected_shape):
        resulting_tensor = fft3d._change_to_channel_last(tf.ones(shape=input_shape))
        assert resulting_tensor.shape == expected_shape

    @pytest.mark.parametrize(
        ("input_shape","expected_context"),[
            (tf.TensorShape((1,3,16,16)), does_not_raise()),
            (tf.TensorShape((1,3,8,16,16)), does_not_raise()),
            (tf.TensorShape((1,3,8)), pytest.raises(ValueError)),
            (tf.TensorShape((1,3,8,16,16,6)), pytest.raises(ValueError)),
        ]
    )
    def test_change_to_channel_last_raises_error(self, fft3d, input_shape, expected_context):
        with expected_context:
            _ = fft3d._change_to_channel_last(tf.ones(shape=input_shape))

class TestiFFT2D(DeepSakiLayerChecks):

    @pytest.mark.parametrize(
        ("input_shape","input_complex", "expected_context"),
        [
            (tf.TensorShape((8, 64, 64, 4)),False, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4)),True, does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3)),False, pytest.raises(ValueError)),
            (tf.TensorShape((1, 32, 32, 3)),True, does_not_raise()),
            (tf.TensorShape((8, 64, 64)),True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64)),True, pytest.raises(ValueError)),
            (tf.TensorShape((8)),True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5)),True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5, 6)),True, pytest.raises(ValueError)),
        ],
    )
    def test_call_raises_error_wrong_input_dim(self,ifft2d,input_shape,input_complex, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_dim(ifft2d,input_shape, expected_context, make_input_complex=input_complex)


    # @pytest.mark.xfail(reason="test not implemented")
    @pytest.mark.parametrize("shift_fft",[True,False])
    @pytest.mark.parametrize(
        ("input_shape","is_channel_first","apply_real_fft","expected_shape"),[
            (tf.TensorShape((1, 16, 16, 3)),False,False,tf.TensorShape((1, 16, 16, 3))),
            (tf.TensorShape((1, 34, 34, 16)),False,False,tf.TensorShape((1, 34, 34, 16))),
            (tf.TensorShape((8, 16, 16, 3)),False,False,tf.TensorShape((8, 16, 16, 3))),
            (tf.TensorShape((1, 5, 16, 16)),True,False,tf.TensorShape((1, 5, 16, 16))),
            (tf.TensorShape((1, 16, 34, 34)),True,False,tf.TensorShape((1, 16, 34, 34))),
            (tf.TensorShape((8, 3, 16, 16)),True,False,tf.TensorShape((8, 3, 16, 16))),
            (tf.TensorShape((1, 16, 9, 3)),False,True,tf.TensorShape((1, 16, 16, 3))),
            (tf.TensorShape((1, 34, 18, 16)),False,True,tf.TensorShape((1, 34, 34, 16))),
            (tf.TensorShape((8, 16, 9, 3)),False,True,tf.TensorShape((8, 16, 16, 3))),
            (tf.TensorShape((1, 5, 16, 9)),True,True,tf.TensorShape((1, 5, 16, 16))),
            (tf.TensorShape((1, 16, 34, 18)),True,True,tf.TensorShape((1, 16, 34, 34))),
            (tf.TensorShape((8, 3, 16, 9)),True,True,tf.TensorShape((8, 3, 16, 16))),
        ]
    )
    def test_call_correct_output_shape(self,input_shape, is_channel_first,apply_real_fft,shift_fft,expected_shape):
        layer_instance = iFFT2D(is_channel_first,apply_real_fft,shift_fft)
        CommonLayerChecks.has_call_correct_output_shape(layer_instance,input_shape,expected_shape, make_input_complex=True)

class TestiFFT3D(DeepSakiLayerChecks):
    @pytest.mark.parametrize(
        ("input_shape", "input_complex", "expected_context"),
        [
            (tf.TensorShape((8, 64, 64, 4)),False, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4)),True, does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3)),False, pytest.raises(ValueError)),
            (tf.TensorShape((1, 32, 32, 3)),True, does_not_raise()),
            (tf.TensorShape((8, 64, 64)),True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64)),True, pytest.raises(ValueError)),
            (tf.TensorShape((8)),True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5)),True, does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3, 9)),True, does_not_raise()),
            (tf.TensorShape((8, 64, 64, 4, 5, 6)),True, pytest.raises(ValueError)),
        ],
    )
    def test_call_raises_error_wrong_input_dim(self,ifft3d,input_shape,input_complex, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_dim(ifft3d,input_shape, expected_context, make_input_complex=input_complex)

    @pytest.mark.xfail(reason="test not implemented")
    def test_call_correct_output_shape(self):
        ...

class TestFourierPooling2D(DeepSakiLayerChecks):
    @pytest.mark.parametrize(
        ("input_shape", "expected_context"),
        [
            (tf.TensorShape((8, 64, 64, 4)), does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3)), does_not_raise()),
            (tf.TensorShape((8, 64, 64)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 64)), pytest.raises(ValueError)),
            (tf.TensorShape((8)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5, 6)), pytest.raises(ValueError)),
        ],
    )
    def test_call_raises_error_wrong_input_dim(self,fourier_pooling_2d,input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_dim(fourier_pooling_2d,input_shape, expected_context)

    @pytest.mark.xfail(reason="test not implemented")
    def test_call_correct_output_shape(self):
        ...

class TestrFFT2DFilter(DeepSakiLayerChecks):
    @pytest.mark.parametrize(
        ("input_shape", "expected_context"),
        [
            (tf.TensorShape((8, 64, 64, 4)), does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3)), does_not_raise()),
            (tf.TensorShape((8, 64, 64)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 64)), pytest.raises(ValueError)),
            (tf.TensorShape((8)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5, 6)), pytest.raises(ValueError)),
        ],
    )
    def test_call_raises_error_wrong_input_dim(self,rfft2d_filter,input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_dim(rfft2d_filter,input_shape, expected_context)

    @pytest.mark.xfail(reason="test not implemented")
    def test_call_correct_output_shape(self):
        ...
