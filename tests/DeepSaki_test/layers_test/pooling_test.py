import pytest
import os
import inspect
from contextlib import nullcontext as does_not_raise

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf
import numpy as np

from DeepSaki.layers.pooling import (
    GlobalSumPooling2D,
    LearnedPooling,
)


class TestGlobalSumPooling2D:
    @pytest.mark.parametrize(
        "data_format, expected_context",
        [
            ("channels_last", does_not_raise()),
            ("channels_first", does_not_raise()),
            ("Any string", pytest.raises(ValueError)),
            (None, pytest.raises(ValueError)),
        ],
    )
    def test_init_raises_error_wrong_data_format(self, data_format, expected_context):
        with expected_context:
            _ = GlobalSumPooling2D(data_format=data_format)

    @pytest.mark.parametrize(
        "input_shape, expected_context",
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
        layer = GlobalSumPooling2D()
        with expected_context:
            _ = layer(tf.ones(shape=input_shape))

    @pytest.mark.parametrize(
        "input,data_format, expected_shape",
        [
            (tf.ones(shape=(8, 64, 64, 3)), "channels_last", tf.TensorShape((8, 3))),
            (tf.ones(shape=(5, 32, 32, 4)), "channels_last", tf.TensorShape((5, 4))),
            (tf.ones(shape=(1, 16, 32, 128)), "channels_last", tf.TensorShape((1, 128))),
            (tf.ones(shape=(2, 64, 8, 512)), "channels_last", tf.TensorShape((2, 512))),
            (tf.ones(shape=(8, 3, 64, 64)), "channels_first", tf.TensorShape((8, 3))),
            (tf.ones(shape=(5, 4, 32, 32)), "channels_first", tf.TensorShape((5, 4))),
            (tf.ones(shape=(1, 128, 16, 32)), "channels_first", tf.TensorShape((1, 128))),
            (tf.ones(shape=(2, 512, 64, 8)), "channels_first", tf.TensorShape((2, 512))),
        ],
    )
    def test_call_correct_shape(self, input, data_format, expected_shape):
        layer = GlobalSumPooling2D(data_format=data_format)
        result = layer(input)
        assert result.shape == expected_shape

    @pytest.mark.parametrize(
        "input, expected",
        [
            (tf.ones(shape=(8, 64, 64, 3)), 64 * 64 * tf.ones(shape=(8, 3))),
            (tf.ones(shape=(5, 32, 32, 4)), 32 * 32 * tf.ones(shape=(5, 4))),
            (tf.ones(shape=(1, 16, 32, 128)), 16 * 32 * tf.ones(shape=(1, 128))),
            (tf.ones(shape=(2, 64, 8, 512)), 64 * 8 * tf.ones(shape=(2, 512))),
        ],
    )
    def test_call_correct_output(self, input, expected):
        layer = GlobalSumPooling2D()
        result = layer(input)
        assert tf.math.reduce_all(result.numpy() == pytest.approx(expected.numpy(), 0.01))

    @pytest.mark.parametrize(
        "input, data_format, expected_shape",
        [
            (tf.TensorShape((8, 64, 64, 3)), "channels_last", tf.TensorShape((8, 3))),
            (tf.TensorShape((5, 32, 32, 4)), "channels_last", tf.TensorShape((5, 4))),
            (tf.TensorShape((1, 16, 32, 128)), "channels_last", tf.TensorShape((1, 128))),
            (tf.TensorShape((2, 64, 8, 512)), "channels_last", tf.TensorShape((2, 512))),
            (tf.TensorShape((8, 3, 64, 64)), "channels_first", tf.TensorShape((8, 3))),
            (tf.TensorShape((5, 4, 32, 32)), "channels_first", tf.TensorShape((5, 4))),
            (tf.TensorShape((1, 128, 16, 32)), "channels_first", tf.TensorShape((1, 128))),
            (tf.TensorShape((2, 512, 64, 8)), "channels_first", tf.TensorShape((2, 512))),
        ],
    )
    def test_compute_output_shape_correct_shape(self, input, data_format, expected_shape):
        layer = GlobalSumPooling2D(data_format=data_format)
        result_shape = layer.compute_output_shape(input)
        assert result_shape == expected_shape


class TestLearnedPooling:
    @pytest.mark.parametrize(
        "input_shape, expected_context",
        [
            (tf.TensorShape((8)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 16)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 16, 16)), pytest.raises(ValueError)),
            (tf.TensorShape((4, 16, 16, 4)), does_not_raise()),
            (tf.TensorShape((8, 32, 32, 3)), does_not_raise()),
            (tf.TensorShape((8, 16, 16, 4, 2)), pytest.raises(ValueError)),
            (tf.TensorShape(()), pytest.raises(ValueError)),
        ],
    )
    def test_call_raises_error_wrong_input_dim(self, input_shape, expected_context):
        layer = LearnedPooling()
        with expected_context:
            _ = layer(tf.ones(shape=input_shape))

    @pytest.mark.parametrize(
        "input, pool_size, expected_shape",
        [
            (tf.ones(shape=(8, 64, 64, 3)), 2, tf.TensorShape((8, 32, 32, 3))),
            (tf.ones(shape=(5, 32, 32, 4)), 4, tf.TensorShape((5, 8, 8, 4))),
            (tf.ones(shape=(1, 16, 32, 128)), 8, tf.TensorShape((1, 2, 4, 128))),
            (tf.ones(shape=(2, 16, 16, 3)), 1, tf.TensorShape((2, 16, 16, 3))),
        ],
    )
    def test_call_correct_shape(self, input, pool_size, expected_shape):
        layer = LearnedPooling(pool_size=pool_size)
        result = layer(input)
        assert result.shape == expected_shape
