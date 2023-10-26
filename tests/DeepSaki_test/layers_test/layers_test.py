import pytest
import os
import inspect

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf
import numpy as np

from DeepSaki.layers.layer_composites import (
    Conv2DSplitted,
    Conv2DBlock,
    DenseBlock,
    DownSampleBlock,
    UpSampleBlock,
    ResidualBlock,
    ResBlockDown,
    ResBlockUp,
    ScaleLayer,
    ScalarGatedSelfAttention,
)

from DeepSaki.layers.fourier_layer import (
    FourierConvolution2D,
    FourierFilter2D,
    FFT2D,
    FFT3D,
    iFFT2D,
    iFFT3D,
    FourierPooling2D,
    rFFT2DFilter,
)

from DeepSaki.layers.padding import ReflectionPadding2D

from DeepSaki.layers.pooling import (
    GlobalSumPooling2D,
    LearnedPooling,
)

from DeepSaki.layers.sub_model_composites import (
    Encoder,
    Bottleneck,
    Decoder,
)


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

    @pytest.mark.xfail(reason="Functionality not yet implemented.")
    def test_input_spec_is_defined_in_init(self, layer_object):
        assert layer_object.input_spec is not None, "'tf.keras.layers.InputSpec' is not defined."
