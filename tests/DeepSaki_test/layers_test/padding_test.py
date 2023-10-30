import os

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
from contextlib import nullcontext as does_not_raise

import tensorflow as tf

from DeepSaki.layers.padding import ReflectionPadding2D
from tests.DeepSaki_test.layers_test.layers_test import CommonLayerChecks
from tests.DeepSaki_test.layers_test.layers_test import DeepSakiLayerChecks

class TestReflectionPadding2D(DeepSakiLayerChecks):
    @pytest.fixture()
    def reflection_padding_2d(self):
        return ReflectionPadding2D()

    @pytest.mark.parametrize(
        ("input", "padding", "expected_shape"),
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
        ("input", "padding", "expected_shape"),
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
        ("input_shape", "padding", "expected_shape"),
        [
            (tf.TensorShape((8, 64, 64, 3)), (0, 0), tf.TensorShape((8, 64, 64, 3))),
            (tf.TensorShape((8, 64, 64, 3)), (1, 1), tf.TensorShape((8, 66, 66, 3))),
            (tf.TensorShape((8, 64, 64, 3)), (2, 2), tf.TensorShape((8, 68, 68, 3))),
            (tf.TensorShape((8, 64, 64, 3)), (2, 1), tf.TensorShape((8, 68, 66, 3))),
            (tf.TensorShape((8, 64, 64, 3)), (0, 1), tf.TensorShape((8, 64, 66, 3))),
        ],
    )
    def test_call_correct_output_shape(self, input_shape, padding, expected_shape):
        layer_instance = ReflectionPadding2D(padding=padding)
        CommonLayerChecks.has_call_correct_output_shape(layer_instance, input_shape, expected_shape)

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
    def test_call_raises_error_wrong_input_spec(self, reflection_padding_2d, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(reflection_padding_2d, input_shape, expected_context)
