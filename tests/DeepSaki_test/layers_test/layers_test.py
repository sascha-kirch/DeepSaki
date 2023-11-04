import inspect
import os

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
from abc import ABC
from abc import abstractmethod

import tensorflow as tf

from DeepSaki.layers.fourier_layer import FFT2D
from DeepSaki.layers.fourier_layer import FFT3D
from DeepSaki.layers.fourier_layer import FourierConvolution2D
from DeepSaki.layers.fourier_layer import FourierFilter2D
from DeepSaki.layers.fourier_layer import FourierPooling2D
from DeepSaki.layers.fourier_layer import iFFT2D
from DeepSaki.layers.fourier_layer import iFFT3D
from DeepSaki.layers.fourier_layer import rFFT2DFilter
from DeepSaki.layers.layer_composites import Conv2DBlock
from DeepSaki.layers.layer_composites import Conv2DSplitted
from DeepSaki.layers.layer_composites import DenseBlock
from DeepSaki.layers.layer_composites import DownSampleBlock
from DeepSaki.layers.layer_composites import ResBlockDown
from DeepSaki.layers.layer_composites import ResBlockUp
from DeepSaki.layers.layer_composites import ResidualBlock
from DeepSaki.layers.layer_composites import ScalarGatedSelfAttention
from DeepSaki.layers.layer_composites import ScaleLayer
from DeepSaki.layers.layer_composites import UpSampleBlock
from DeepSaki.layers.padding import ReflectionPadding2D
from DeepSaki.layers.pooling import GlobalSumPooling2D
from DeepSaki.layers.pooling import LearnedPooling
from DeepSaki.layers.sub_model_composites import Bottleneck
from DeepSaki.layers.sub_model_composites import Decoder
from DeepSaki.layers.sub_model_composites import Encoder


class DeepSakiLayerChecks(ABC):
    @abstractmethod
    def test_call_raises_error_wrong_input_spec(self):
        ...

    @abstractmethod
    def test_call_correct_output_shape(self):
        ...


class CommonLayerChecks:
    @staticmethod
    def has_call_correct_output_shape(layer_instance, input_shape, expected_shape, make_input_complex=False):
        input = tf.ones(shape=input_shape)
        if make_input_complex:
            input = tf.complex(real=input, imag=input)
        output = layer_instance(input)
        assert output.shape == expected_shape

    @staticmethod
    def does_call_raises_error_wrong_input_spec(
        layer_instance, input_shape, expected_context, make_input_complex=False
    ):
        with expected_context:
            input = tf.ones(shape=input_shape)
            if make_input_complex:
                input = tf.complex(real=input, imag=input)
            _ = layer_instance(input)


@pytest.mark.parametrize(
    "layer_object",
    [
        Conv2DSplitted(),
        Conv2DBlock(),
        DenseBlock(units=1),
        DownSampleBlock(),
        UpSampleBlock(),
        ResidualBlock(),
        ResBlockDown(),
        ResBlockUp(),
        ScaleLayer(),
        ScalarGatedSelfAttention(),
        FourierConvolution2D(),
        FourierFilter2D(),
        FFT2D(),
        FFT3D(),
        iFFT2D(),
        iFFT3D(),
        ReflectionPadding2D(),
        FourierPooling2D(),
        rFFT2DFilter(),
        GlobalSumPooling2D(),
        LearnedPooling(),
        Encoder(),
        Bottleneck(),
        Decoder(),
    ],
)
class TestGenericLayer:
    def test_get_dict(self, layer_object):
        config = layer_object.get_config()
        # func bellow gets all variable names of the __init__ param list. [1::] removes "self" from that list.
        expected_keys = inspect.getfullargspec(layer_object.__init__)[0][1::]
        key_in_config = [key in config for key in expected_keys]
        assert all(key_in_config), f"not all expected keys found in config: {key_in_config}"

    def test_layer_is_subclass_of_tensorflow_layer(self, layer_object):
        assert isinstance(layer_object, tf.keras.layers.Layer)

    def test_input_spec_is_defined_in_init(self, layer_object):
        assert layer_object.input_spec is not None, "'tf.keras.layers.InputSpec' is not defined."
