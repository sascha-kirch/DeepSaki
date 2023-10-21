import pytest
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf
import numpy as np

from DeepSaki.layers.layer_helper import PaddingType, InitializerFunc, get_initializer, pad_func, dropout_func, plot_layer
from DeepSaki.initializers.he_alpha import HeAlphaNormal, HeAlphaUniform
from DeepSaki.layers.padding import ReflectionPadding2D

class TestGetInitializerFunc():

    @pytest.mark.parametrize(
        "initializer_func, expected",
        [
            (InitializerFunc.RANDOM_NORMAL, tf.keras.initializers.RandomNormal),
            (InitializerFunc.RANDOM_UNIFORM, tf.keras.initializers.RandomUniform),
            (InitializerFunc.GLOROT_NORMAL, tf.keras.initializers.GlorotNormal),
            (InitializerFunc.GLOROT_UNIFORM, tf.keras.initializers.GlorotUniform),
            (InitializerFunc.HE_NORMAL, tf.keras.initializers.HeNormal),
            (InitializerFunc.HE_UNIFORM, tf.keras.initializers.HeUniform),
            (InitializerFunc.HE_ALPHA_NORMAL, HeAlphaNormal),
            (InitializerFunc.HE_ALPHA_UNIFORM, HeAlphaUniform),
        ],
    )
    def test_get_initializer_func_returns_correct_initializer(self, initializer_func, expected):
        returned_initializer = get_initializer(initializer_func)
        assert isinstance(returned_initializer, expected)

    @pytest.mark.parametrize(
        "initializer_func",
        [
            InitializerFunc.NONE,
            None,
            "Any_string"
        ],
    )
    def test_get_initializer_func_raises_value_error(self, initializer_func):
            with pytest.raises(ValueError):
                _ = get_initializer(initializer_func)

class TestPadFunc():
    @pytest.mark.parametrize(
        "padding_type, expected",
        [
            (PaddingType.ZERO, tf.keras.layers.ZeroPadding2D),
            (PaddingType.REFLECTION, ReflectionPadding2D),
        ],
    )
    def test_pad_func_returns_correct_padding(self, padding_type, expected):
        returned_padding = pad_func(padding_type=padding_type)
        assert isinstance(returned_padding, expected)

    @pytest.mark.parametrize(
        "padding_type",
        [
            PaddingType.NONE,
            None,
            "Any_string"
        ],
    )
    def test_pad_func_raises_value_error(self, padding_type):
            with pytest.raises(ValueError):
                _ = pad_func(padding_type=padding_type)

@pytest.mark.parametrize("input_filters, expected",[
    (1, tf.keras.layers.Dropout),
    (2, tf.keras.layers.SpatialDropout2D),
    (64, tf.keras.layers.SpatialDropout2D),
    (99, tf.keras.layers.SpatialDropout2D),
])
def test_dropout_func_returns_correct_dropout(input_filters, expected):
    output = dropout_func(filters=input_filters, dropout_rate=0.5)
    assert isinstance(output,expected)

@pytest.mark.parametrize("input_filters, error_type",[
    (0, ValueError),
    (-10, ValueError),
    (-99, ValueError),
    ("Any String", TypeError),
    (None, TypeError),
    (3.0, TypeError),
])
def test_dropout_func_raises_error(input_filters, error_type):
    with pytest.raises(error_type):
        _ = dropout_func(filters=input_filters, dropout_rate=0.5)
