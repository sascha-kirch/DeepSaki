import os

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
from contextlib import nullcontext as does_not_raise

import tensorflow as tf

from DeepSaki.layers.fourier_layer import FFT2D
from DeepSaki.layers.fourier_layer import FFT3D
from DeepSaki.layers.fourier_layer import FourierConvolution2D
from DeepSaki.layers.fourier_layer import FourierFilter2D
from DeepSaki.layers.fourier_layer import FourierLayer
from DeepSaki.layers.fourier_layer import FourierPooling2D
from DeepSaki.layers.fourier_layer import FrequencyFilter
from DeepSaki.layers.fourier_layer import MultiplicationType
from DeepSaki.layers.fourier_layer import iFFT2D
from DeepSaki.layers.fourier_layer import iFFT3D
from DeepSaki.layers.fourier_layer import rFFT2DFilter
from tests.DeepSaki_test.layers_test.layers_test import CommonLayerChecks
from tests.DeepSaki_test.layers_test.layers_test import DeepSakiLayerChecks

class TestFourierLayer:
    @pytest.fixture()
    def fourier_layer(self):
        return FourierLayer()

    @pytest.mark.parametrize(
        ("input_shape", "expected_shape"),
        [
            (tf.TensorShape((1, 16, 16, 3)), tf.TensorShape((1, 3, 16, 16))),
            (tf.TensorShape((1, 8, 16, 16, 3)), tf.TensorShape((1, 3, 8, 16, 16))),
        ],
    )
    def test_change_to_channel_first_correct_shape(self, fourier_layer, input_shape, expected_shape):
        resulting_tensor = fourier_layer._change_to_channel_first(tf.ones(shape=input_shape))
        assert resulting_tensor.shape == expected_shape

    @pytest.mark.parametrize(
        ("input_shape", "expected_context"),
        [
            (tf.TensorShape((1, 16, 16, 3)), does_not_raise()),
            (tf.TensorShape((1, 8, 16, 16, 3)), does_not_raise()),
            (tf.TensorShape((1, 8, 16)), pytest.raises(ValueError)),
            (tf.TensorShape((1, 8, 16, 16, 3, 6)), pytest.raises(ValueError)),
        ],
    )
    def test_change_to_channel_first_raises_error(self, fourier_layer, input_shape, expected_context):
        with expected_context:
            _ = fourier_layer._change_to_channel_first(tf.ones(shape=input_shape))

    @pytest.mark.parametrize(
        ("input_shape", "expected_shape"),
        [
            (tf.TensorShape((1, 3, 16, 16)), tf.TensorShape((1, 16, 16, 3))),
            (tf.TensorShape((1, 3, 8, 16, 16)), tf.TensorShape((1, 8, 16, 16, 3))),
        ],
    )
    def test_change_to_channel_last_correct_shape(self, fourier_layer, input_shape, expected_shape):
        resulting_tensor = fourier_layer._change_to_channel_last(tf.ones(shape=input_shape))
        assert resulting_tensor.shape == expected_shape

    @pytest.mark.parametrize(
        ("input_shape", "expected_context"),
        [
            (tf.TensorShape((1, 3, 16, 16)), does_not_raise()),
            (tf.TensorShape((1, 3, 8, 16, 16)), does_not_raise()),
            (tf.TensorShape((1, 3, 8)), pytest.raises(ValueError)),
            (tf.TensorShape((1, 3, 8, 16, 16, 6)), pytest.raises(ValueError)),
        ],
    )
    def test_change_to_channel_last_raises_error(self, fourier_layer, input_shape, expected_context):
        with expected_context:
            _ = fourier_layer._change_to_channel_last(tf.ones(shape=input_shape))

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
        fourier_layer,
        batch,
        height,
        width,
        channel,
        filter,
    ):
        image_tensor = tf.ones(shape=(batch, channel, height, width))
        kernel_tensor = tf.ones(shape=(filter, channel, height, width))
        result = fourier_layer._elementwise_product(image_tensor, kernel_tensor)
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
        fourier_layer,
        batch,
        height,
        width,
        channel,
        filter,
    ):
        image_tensor = tf.ones(shape=(batch, channel, height, width))
        kernel_tensor = tf.ones(shape=(filter, channel, height, width))
        result = fourier_layer._elementwise_product(image_tensor, kernel_tensor)
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
        fourier_layer,
        batch,
        height,
        width,
        channel,
        filter,
    ):
        image_tensor = tf.ones(shape=(batch, channel, height, width))
        kernel_tensor = tf.ones(shape=(filter, channel, height, width))
        result_element_wise = fourier_layer._elementwise_product(image_tensor, kernel_tensor)
        result_matrix_product = fourier_layer._matrix_product(image_tensor, kernel_tensor)
        assert tf.math.reduce_all(result_element_wise.numpy() == pytest.approx(result_matrix_product.numpy(), 0.01))

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
    def test_get_multiplication_function_raises_error_wrong_multiplication_type(
        self, fourier_layer, multiplication_type, expected_context
    ):
        with expected_context:
            _ = fourier_layer._get_multiplication_function(multiplication_type=multiplication_type)


class TestFourierConvolution2D(DeepSakiLayerChecks):
    @pytest.fixture()
    def fourier_convolution_2D(self):
        return FourierConvolution2D()

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
    def test_get_image_padding(self, fourier_convolution_2D, kernels, expected):
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
    def test_fill_image_shape_power_2(self, fourier_convolution_2D, input_shape, expected_shape):
        output_shape = fourier_convolution_2D._fill_image_shape_power_2(input_shape)
        assert output_shape == expected_shape

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
    def test_call_raises_error_wrong_input_spec(self, fourier_convolution_2D, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(fourier_convolution_2D, input_shape, expected_context)

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
            filters,
            kernels,
            use_bias,
            is_channel_first,
            apply_conjugate,
            pad_to_power_2,
            method,
        )
        CommonLayerChecks.has_call_correct_output_shape(layer_instance, input_shape, expected_shape)


class TestFourierFilter2D(DeepSakiLayerChecks):
    @pytest.fixture()
    def fourier_filter_2D(self):
        return FourierFilter2D()

    @pytest.mark.parametrize(
        ("input_shape", "input_complex", "expected_context"),
        [
            (tf.TensorShape((8, 64, 64, 4)), False, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4)), True, does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3)), False, pytest.raises(ValueError)),
            (tf.TensorShape((1, 32, 32, 3)), True, does_not_raise()),
            (tf.TensorShape((8, 64, 64)), True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64)), True, pytest.raises(ValueError)),
            (tf.TensorShape((8)), True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5)), True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5, 6)), True, pytest.raises(ValueError)),
        ],
    )
    def test_call_raises_error_wrong_input_spec(self, fourier_filter_2D, input_shape, input_complex, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(
            fourier_filter_2D, input_shape, expected_context, make_input_complex=input_complex
        )

    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize(
        "multiplication_type", [MultiplicationType.ELEMENT_WISE, MultiplicationType.MATRIX_PRODUCT]
    )
    @pytest.mark.parametrize(
        ("input_shape", "filters", "is_channel_first", "expected_shape"),
        [
            (tf.TensorShape((1, 8, 8, 3)), 12, False, tf.TensorShape((1, 8, 8, 12))),
            (tf.TensorShape((8, 3, 8, 8)), 9, True, tf.TensorShape((8, 9, 8, 8))),
        ],
    )
    def test_call_correct_output_shape(
        self, input_shape, filters, is_channel_first, use_bias, multiplication_type, expected_shape
    ):
        layer_instance = FourierFilter2D(filters, use_bias, is_channel_first, multiplication_type)
        CommonLayerChecks.has_call_correct_output_shape(
            layer_instance, input_shape, expected_shape, make_input_complex=True
        )


class TestFFT2D(DeepSakiLayerChecks):
    @pytest.fixture()
    def fft2d(self):
        return FFT2D()

    @pytest.mark.parametrize(
        ("input_shape", "input_complex", "apply_real_fft", "expected_context"),
        [
            (tf.TensorShape((8, 64, 64, 4)), True, False, does_not_raise()),
            (tf.TensorShape((8, 64, 64, 4)), False, False, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4)), True, True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4)), False, True, does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3)), True, False, does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3)), False, False, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64)), True, False, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64)), True, False, pytest.raises(ValueError)),
            (tf.TensorShape((8)), True, False, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5)), True, False, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5, 6)), True, False, pytest.raises(ValueError)),
        ],
    )
    def test_call_raises_error_wrong_input_spec(self, input_shape, input_complex, apply_real_fft, expected_context):
        layer_instance = FFT2D(apply_real_fft=apply_real_fft)
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(
            layer_instance, input_shape, expected_context, make_input_complex=input_complex
        )

    @pytest.mark.parametrize("shift_fft", [True, False])
    @pytest.mark.parametrize(
        ("input_shape", "is_channel_first", "apply_real_fft", "expected_shape"),
        [
            (tf.TensorShape((1, 16, 8, 3)), False, False, tf.TensorShape((1, 16, 8, 3))),
            (tf.TensorShape((1, 16, 16, 3)), False, False, tf.TensorShape((1, 16, 16, 3))),
            (tf.TensorShape((1, 34, 34, 16)), False, False, tf.TensorShape((1, 34, 34, 16))),
            (tf.TensorShape((8, 12, 16, 3)), False, False, tf.TensorShape((8, 12, 16, 3))),
            (tf.TensorShape((1, 5, 16, 16)), True, False, tf.TensorShape((1, 5, 16, 16))),
            (tf.TensorShape((1, 16, 34, 34)), True, False, tf.TensorShape((1, 16, 34, 34))),
            (tf.TensorShape((8, 3, 16, 16)), True, False, tf.TensorShape((8, 3, 16, 16))),
            (tf.TensorShape((1, 16, 16, 3)), False, True, tf.TensorShape((1, 16, 9, 3))),
            (tf.TensorShape((1, 34, 18, 16)), False, True, tf.TensorShape((1, 34, 10, 16))),
            (tf.TensorShape((8, 16, 16, 3)), False, True, tf.TensorShape((8, 16, 9, 3))),
            (tf.TensorShape((1, 5, 16, 16)), True, True, tf.TensorShape((1, 5, 16, 9))),
            (tf.TensorShape((1, 16, 34, 34)), True, True, tf.TensorShape((1, 16, 34, 18))),
            (tf.TensorShape((8, 3, 16, 16)), True, True, tf.TensorShape((8, 3, 16, 9))),
        ],
    )
    def test_call_correct_output_shape(self, input_shape, is_channel_first, apply_real_fft, shift_fft, expected_shape):
        layer_instance = FFT2D(is_channel_first, apply_real_fft, shift_fft)
        CommonLayerChecks.has_call_correct_output_shape(
            layer_instance, input_shape, expected_shape, make_input_complex=(not apply_real_fft)
        )


class TestFFT3D(DeepSakiLayerChecks):
    @pytest.fixture()
    def fft3d(self):
        return FFT3D()

    @pytest.mark.parametrize(
        ("input_shape", "input_complex", "apply_real_fft", "expected_context"),
        [
            (tf.TensorShape((8, 64, 64, 4)), True, False, does_not_raise()),
            (tf.TensorShape((8, 64, 64, 4)), False, False, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4)), True, True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4)), False, True, does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3)), True, False, does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3)), False, False, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64)), True, False, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64)), True, False, pytest.raises(ValueError)),
            (tf.TensorShape((8)), True, False, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5)), True, False, does_not_raise()),
            (tf.TensorShape((8, 64, 64, 4, 5)), False, False, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5)), True, True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5)), False, True, does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3, 9)), True, False, does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3, 9)), False, False, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5, 6)), True, False, pytest.raises(ValueError)),
        ],
    )
    def test_call_raises_error_wrong_input_spec(self, input_shape, input_complex, apply_real_fft, expected_context):
        layer_instance = FFT3D(apply_real_fft=apply_real_fft)
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(
            layer_instance, input_shape, expected_context, make_input_complex=input_complex
        )

    @pytest.mark.parametrize("shift_fft", [True, False])
    @pytest.mark.parametrize(
        ("input_shape", "is_channel_first", "apply_real_fft", "expected_shape"),
        [
            (tf.TensorShape((1, 16, 16, 3)), False, False, tf.TensorShape((1, 16, 16, 3))),
            (tf.TensorShape((8, 34, 34, 16)), False, False, tf.TensorShape((8, 34, 34, 16))),
            (tf.TensorShape((1, 5, 16, 16)), True, False, tf.TensorShape((1, 5, 16, 16))),
            (tf.TensorShape((8, 16, 34, 34)), True, False, tf.TensorShape((8, 16, 34, 34))),
            (tf.TensorShape((1, 16, 16, 3)), False, True, tf.TensorShape((1, 16, 9, 3))),
            (tf.TensorShape((8, 34, 34, 16)), False, True, tf.TensorShape((8, 34, 18, 16))),
            (tf.TensorShape((1, 5, 16, 16)), True, True, tf.TensorShape((1, 5, 16, 9))),
            (tf.TensorShape((8, 16, 34, 34)), True, True, tf.TensorShape((8, 16, 34, 18))),
            (tf.TensorShape((1, 8, 16, 16, 3)), False, False, tf.TensorShape((1, 8, 16, 16, 3))),
            (tf.TensorShape((8, 1, 34, 34, 16)), False, False, tf.TensorShape((8, 1, 34, 34, 16))),
            (tf.TensorShape((1, 5, 8, 16, 16)), True, False, tf.TensorShape((1, 5, 8, 16, 16))),
            (tf.TensorShape((8, 16, 1, 34, 34)), True, False, tf.TensorShape((8, 16, 1, 34, 34))),
            (tf.TensorShape((1, 8, 16, 16, 3)), False, True, tf.TensorShape((1, 8, 16, 9, 3))),
            (tf.TensorShape((8, 1, 34, 34, 16)), False, True, tf.TensorShape((8, 1, 34, 18, 16))),
            (tf.TensorShape((1, 5, 8, 16, 16)), True, True, tf.TensorShape((1, 5, 8, 16, 9))),
            (tf.TensorShape((8, 16, 1, 34, 34)), True, True, tf.TensorShape((8, 16, 1, 34, 18))),
        ],
    )
    def test_call_correct_output_shape(self, input_shape, is_channel_first, apply_real_fft, shift_fft, expected_shape):
        layer_instance = FFT3D(is_channel_first, apply_real_fft, shift_fft)
        CommonLayerChecks.has_call_correct_output_shape(
            layer_instance, input_shape, expected_shape, make_input_complex=(not apply_real_fft)
        )


class TestiFFT2D(DeepSakiLayerChecks):
    @pytest.fixture()
    def ifft2d(self):
        return iFFT2D()

    @pytest.mark.parametrize(
        ("input_shape", "input_complex", "expected_context"),
        [
            (tf.TensorShape((8, 64, 64, 4)), False, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4)), True, does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3)), False, pytest.raises(ValueError)),
            (tf.TensorShape((1, 32, 32, 3)), True, does_not_raise()),
            (tf.TensorShape((8, 64, 64)), True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64)), True, pytest.raises(ValueError)),
            (tf.TensorShape((8)), True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5)), True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5, 6)), True, pytest.raises(ValueError)),
        ],
    )
    def test_call_raises_error_wrong_input_spec(self, ifft2d, input_shape, input_complex, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(
            ifft2d, input_shape, expected_context, make_input_complex=input_complex
        )

    @pytest.mark.parametrize("shift_fft", [True, False])
    @pytest.mark.parametrize(
        ("input_shape", "is_channel_first", "apply_real_fft", "fft_length", "expected_shape"),
        [
            (tf.TensorShape((1, 16, 16, 3)), False, False, None, tf.TensorShape((1, 16, 16, 3))),
            (tf.TensorShape((1, 34, 34, 16)), False, False, (10, 10), tf.TensorShape((1, 34, 34, 16))),
            (tf.TensorShape((8, 16, 16, 3)), False, False, None, tf.TensorShape((8, 16, 16, 3))),
            (tf.TensorShape((1, 5, 16, 16)), True, False, None, tf.TensorShape((1, 5, 16, 16))),
            (tf.TensorShape((1, 16, 34, 34)), True, False, None, tf.TensorShape((1, 16, 34, 34))),
            (tf.TensorShape((8, 3, 16, 16)), True, False, None, tf.TensorShape((8, 3, 16, 16))),
            (tf.TensorShape((1, 16, 9, 3)), False, True, None, tf.TensorShape((1, 16, 16, 3))),
            (tf.TensorShape((1, 34, 18, 16)), False, True, None, tf.TensorShape((1, 34, 34, 16))),
            (tf.TensorShape((8, 16, 9, 3)), False, True, (32, 32), tf.TensorShape((8, 32, 32, 3))),
            (tf.TensorShape((1, 5, 16, 9)), True, True, None, tf.TensorShape((1, 5, 16, 16))),
            (tf.TensorShape((1, 16, 34, 18)), True, True, None, tf.TensorShape((1, 16, 34, 34))),
            (tf.TensorShape((8, 3, 16, 9)), True, True, (20, 20), tf.TensorShape((8, 3, 20, 20))),
        ],
    )
    def test_call_correct_output_shape(
        self, input_shape, is_channel_first, apply_real_fft, shift_fft, fft_length, expected_shape
    ):
        layer_instance = iFFT2D(is_channel_first, apply_real_fft, shift_fft, fft_length)
        CommonLayerChecks.has_call_correct_output_shape(
            layer_instance, input_shape, expected_shape, make_input_complex=True
        )


class TestiFFT3D(DeepSakiLayerChecks):
    @pytest.fixture()
    def ifft3d(self):
        return iFFT3D()

    @pytest.mark.parametrize(
        ("input_shape", "input_complex", "expected_context"),
        [
            (tf.TensorShape((8, 64, 64, 4)), False, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4)), True, does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3)), False, pytest.raises(ValueError)),
            (tf.TensorShape((1, 32, 32, 3)), True, does_not_raise()),
            (tf.TensorShape((8, 64, 64)), True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64)), True, pytest.raises(ValueError)),
            (tf.TensorShape((8)), True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5)), True, does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3, 9)), True, does_not_raise()),
            (tf.TensorShape((8, 64, 64, 4, 5, 6)), True, pytest.raises(ValueError)),
        ],
    )
    def test_call_raises_error_wrong_input_spec(self, ifft3d, input_shape, input_complex, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(
            ifft3d, input_shape, expected_context, make_input_complex=input_complex
        )

    @pytest.mark.parametrize("shift_fft", [True, False])
    @pytest.mark.parametrize(
        ("input_shape", "is_channel_first", "apply_real_fft", "fft_length", "expected_shape"),
        [
            (tf.TensorShape((1, 16, 16, 3)), False, False, None, tf.TensorShape((1, 16, 16, 3))),
            (tf.TensorShape((8, 34, 34, 16)), False, False, (10, 20, 30), tf.TensorShape((8, 34, 34, 16))),
            (tf.TensorShape((1, 5, 16, 16)), True, False, None, tf.TensorShape((1, 5, 16, 16))),
            (tf.TensorShape((8, 16, 34, 34)), True, False, None, tf.TensorShape((8, 16, 34, 34))),
            (tf.TensorShape((1, 16, 9, 3)), False, True, None, tf.TensorShape((1, 16, 16, 3))),
            (tf.TensorShape((8, 34, 18, 16)), False, True, None, tf.TensorShape((8, 34, 34, 16))),
            (tf.TensorShape((1, 5, 16, 9)), True, True, (10, 20, 30), tf.TensorShape((1, 10, 20, 30))),
            (tf.TensorShape((8, 16, 34, 18)), True, True, None, tf.TensorShape((8, 16, 34, 34))),
            (tf.TensorShape((1, 8, 16, 16, 3)), False, False, None, tf.TensorShape((1, 8, 16, 16, 3))),
            (tf.TensorShape((8, 1, 34, 34, 16)), False, False, None, tf.TensorShape((8, 1, 34, 34, 16))),
            (tf.TensorShape((1, 5, 8, 16, 16)), True, False, None, tf.TensorShape((1, 5, 8, 16, 16))),
            (tf.TensorShape((8, 16, 1, 34, 34)), True, False, None, tf.TensorShape((8, 16, 1, 34, 34))),
            (tf.TensorShape((1, 8, 16, 9, 3)), False, True, None, tf.TensorShape((1, 8, 16, 16, 3))),
            (tf.TensorShape((8, 1, 34, 18, 16)), False, True, (10, 20, 30), tf.TensorShape((8, 10, 20, 30, 16))),
            (tf.TensorShape((1, 5, 8, 16, 9)), True, True, None, tf.TensorShape((1, 5, 8, 16, 16))),
            (tf.TensorShape((8, 16, 1, 34, 18)), True, True, None, tf.TensorShape((8, 16, 1, 34, 34))),
        ],
    )
    def test_call_correct_output_shape(
        self, input_shape, is_channel_first, apply_real_fft, shift_fft, fft_length, expected_shape
    ):
        layer_instance = iFFT3D(is_channel_first, apply_real_fft, shift_fft, fft_length)
        CommonLayerChecks.has_call_correct_output_shape(
            layer_instance, input_shape, expected_shape, make_input_complex=True
        )


class TestFourierPooling2D(DeepSakiLayerChecks):
    @pytest.fixture()
    def fourier_pooling_2d(self):
        return FourierPooling2D()

    @pytest.mark.parametrize(
        ("input_shape", "input_complex", "expected_context"),
        [
            (tf.TensorShape((8, 64, 64, 4)), False, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4)), True, does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3)), False, pytest.raises(ValueError)),
            (tf.TensorShape((1, 32, 32, 3)), True, does_not_raise()),
            (tf.TensorShape((8, 64, 64)), True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64)), True, pytest.raises(ValueError)),
            (tf.TensorShape((8)), True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5)), True, pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5, 6)), True, pytest.raises(ValueError)),
        ],
    )
    def test_call_raises_error_wrong_input_spec(self, fourier_pooling_2d, input_shape, input_complex, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(
            fourier_pooling_2d, input_shape, expected_context, make_input_complex=input_complex
        )

    @pytest.mark.parametrize(
        ("input_shape", "is_channel_first", "input_from_rfft", "expected_shape"),
        [
            (tf.TensorShape((1, 16, 16, 3)), False, False, tf.TensorShape((1, 8, 8, 3))),
            (tf.TensorShape((8, 34, 34, 9)), False, False, tf.TensorShape((8, 17, 17, 9))),
            (tf.TensorShape((1, 16, 9, 3)), False, True, tf.TensorShape((1, 8, 5, 3))),
            (tf.TensorShape((8, 34, 18, 7)), False, True, tf.TensorShape((8, 17, 9, 7))),
            (tf.TensorShape((1, 3, 16, 16)), True, False, tf.TensorShape((1, 3, 8, 8))),
            (tf.TensorShape((8, 9, 34, 34)), True, False, tf.TensorShape((8, 9, 17, 17))),
            (tf.TensorShape((1, 3, 16, 9)), True, True, tf.TensorShape((1, 3, 8, 5))),
            (tf.TensorShape((8, 7, 34, 18)), True, True, tf.TensorShape((8, 7, 17, 9))),
        ],
    )
    def test_call_correct_output_shape(self, input_shape, is_channel_first, input_from_rfft, expected_shape):
        layer_instance = FourierPooling2D(is_channel_first, input_from_rfft)
        CommonLayerChecks.has_call_correct_output_shape(
            layer_instance, input_shape, expected_shape, make_input_complex=True
        )


class TestrFFT2DFilter(DeepSakiLayerChecks):
    @pytest.fixture()
    def rfft2d_filter(self):
        return rFFT2DFilter()

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
    def test_call_raises_error_wrong_input_spec(self, rfft2d_filter, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(rfft2d_filter, input_shape, expected_context)

    @pytest.mark.parametrize("filter_type", [FrequencyFilter.LOW_PASS, FrequencyFilter.HIGH_PASS])
    @pytest.mark.parametrize(
        ("input_shape", "is_channel_first"),
        [
            (tf.TensorShape((1, 16, 16, 3)), False),
            (tf.TensorShape((1, 3, 30, 30)), True),
            (tf.TensorShape((8, 32, 32, 9)), False),
            (tf.TensorShape((8, 9, 25, 25)), True),
        ],
    )
    def test_call_correct_output_shape(self, input_shape, is_channel_first, filter_type):
        layer_instance = rFFT2DFilter(is_channel_first, filter_type)
        CommonLayerChecks.has_call_correct_output_shape(layer_instance, input_shape, input_shape)
