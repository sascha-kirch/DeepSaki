import pytest
import os
import inspect

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf
import numpy as np
from contextlib import nullcontext as does_not_raise

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


@pytest.fixture(scope="class")
def layer_object():
    return FourierConvolution2D()


class TestFourierConvolution2D:
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
        batch,
        height,
        width,
        channel,
        filter,
    ):
        layer_object = FourierConvolution2D()
        image_tensor = tf.ones(shape=(batch, channel, height, width))
        kernel_tensor = tf.ones(shape=(filter, channel, height, width))
        result = layer_object._elementwise_product(image_tensor, kernel_tensor)
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
        batch,
        height,
        width,
        channel,
        filter,
    ):
        layer_object = FourierConvolution2D()
        image_tensor = tf.ones(shape=(batch, channel, height, width))
        kernel_tensor = tf.ones(shape=(filter, channel, height, width))
        result = layer_object._elementwise_product(image_tensor, kernel_tensor)
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
        batch,
        height,
        width,
        channel,
        filter,
    ):
        layer_object = FourierConvolution2D()
        image_tensor = tf.ones(shape=(batch, channel, height, width))
        kernel_tensor = tf.ones(shape=(filter, channel, height, width))
        result_element_wise = layer_object._elementwise_product(image_tensor, kernel_tensor)
        result_matrix_product = layer_object._matrix_product(image_tensor, kernel_tensor)
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
    def test_get_image_padding(self, kernels, expected):
        layer_object = FourierConvolution2D(kernels=kernels)
        padding = layer_object._get_image_padding()
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
    def test_fill_image_shape_power_2(self, input_shape, expected_shape):
        layer_object = FourierConvolution2D()
        output_shape = layer_object._fill_image_shape_power_2(input_shape)
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
    def test_call_raises_error_wrong_input(self, input_shape, expected_context):
        layer = FourierConvolution2D()
        with expected_context:
            _ = layer(tf.ones(shape=input_shape))

    @pytest.mark.parametrize("kernels", [(3, 3), (9, 9), (16, 16)])
    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize("apply_conjugate", [True, False])
    @pytest.mark.parametrize("pad_to_power_2", [True, False])
    @pytest.mark.parametrize("method", [MultiplicationType.ELEMENT_WISE, MultiplicationType.MATRIX_PRODUCT])
    @pytest.mark.parametrize(
        ("input", "filters", "is_channel_first", "expected_shape"),
        [
            (tf.ones(shape=(1, 16, 16, 3)), 5, False, tf.TensorShape((1, 16, 16, 5))),
            (tf.ones(shape=(1, 34, 30, 16)), 9, False, tf.TensorShape((1, 34, 30, 9))),
            (tf.ones(shape=(8, 16, 16, 3)), 3, False, tf.TensorShape((8, 16, 16, 3))),
            (tf.ones(shape=(1, 5, 16, 16)), 5, True, tf.TensorShape((1, 5, 16, 16))),
            (tf.ones(shape=(1, 16, 17, 34)), 9, True, tf.TensorShape((1, 9, 17, 34))),
            (tf.ones(shape=(8, 3, 16, 16)), 3, True, tf.TensorShape((8, 3, 16, 16))),
        ],
    )
    def test_call_correct_shape(
        self,
        input,
        filters,
        kernels,
        use_bias,
        is_channel_first,
        apply_conjugate,
        pad_to_power_2,
        method,
        expected_shape,
    ):
        layer = FourierConvolution2D(
            filters, kernels, use_bias, is_channel_first, apply_conjugate, pad_to_power_2, method
        )
        output = layer(input)
        assert output.shape == expected_shape


class TestFFT2D:
    @pytest.mark.xfail(reason="test not implemented")
    def test_call_raises_error_wrong_input(self):
        ...

    @pytest.mark.xfail(reason="test not implemented")
    def test_call_correct_shape(self):
        ...
