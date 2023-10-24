import pytest
import os
import inspect

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf
import numpy as np

from DeepSaki.layers.padding import ReflectionPadding2D


class TestReflectionPadding2D:
    @pytest.mark.parametrize(
        "input, padding, expected_shape",
        [
            (tf.ones(shape=(8, 64, 64, 3)), (0, 0), tf.TensorShape((8, 64, 64, 3))),
            (tf.ones(shape=(8, 64, 64, 3)), (1, 1), tf.TensorShape((8, 66, 66, 3))),
            (tf.ones(shape=(8, 64, 64, 3)), (2, 2), tf.TensorShape((8, 68, 68, 3))),
            (tf.ones(shape=(8, 64, 64, 3)), (2, 1), tf.TensorShape((8, 68, 66, 3))),
            (tf.ones(shape=(8, 64, 64, 3)), (0, 1), tf.TensorShape((8, 64, 66, 3))),
        ],
    )
    def test_padding_func_correct_shape(self, input, padding, expected_shape):
        layer = ReflectionPadding2D(padding=padding)
        output = layer._padding_func(input)
        assert output.shape == expected_shape

    @pytest.mark.parametrize(
        "input, padding, expected_shape",
        [
            (tf.TensorShape((8, 64, 64, 3)), (0, 0), tf.TensorShape((8, 64, 64, 3))),
            (tf.TensorShape((8, 64, 64, 3)), (1, 1), tf.TensorShape((8, 66, 66, 3))),
            (tf.TensorShape((8, 64, 64, 3)), (2, 2), tf.TensorShape((8, 68, 68, 3))),
            (tf.TensorShape((8, 64, 64, 3)), (2, 1), tf.TensorShape((8, 68, 66, 3))),
            (tf.TensorShape((8, 64, 64, 3)), (0, 1), tf.TensorShape((8, 64, 66, 3))),
        ],
    )
    def test_compute_outputshape_correct_shape(self, input, padding, expected_shape):
        layer = ReflectionPadding2D(padding=padding)
        output = layer.compute_output_shape(input)
        assert output == expected_shape

    @pytest.mark.parametrize(
        "input, padding, expected_shape",
        [
            (tf.ones(shape=(8, 64, 64, 3)), (0, 0), tf.TensorShape((8, 64, 64, 3))),
            (tf.ones(shape=(8, 64, 64, 3)), (1, 1), tf.TensorShape((8, 66, 66, 3))),
            (tf.ones(shape=(8, 64, 64, 3)), (2, 2), tf.TensorShape((8, 68, 68, 3))),
            (tf.ones(shape=(8, 64, 64, 3)), (2, 1), tf.TensorShape((8, 68, 66, 3))),
            (tf.ones(shape=(8, 64, 64, 3)), (0, 1), tf.TensorShape((8, 64, 66, 3))),
        ],
    )
    def test_call_correct_shape(self, input, padding, expected_shape):
        layer = ReflectionPadding2D(padding=padding)
        output = layer(input)
        assert output.shape == expected_shape
