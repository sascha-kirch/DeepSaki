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
    ScalarGatedSelfAttention,)



class TestScaleLayer:

    @pytest.mark.skip(reason="Not implemented yet.")
    def test_init(self):
        ...

    @pytest.mark.skip(reason="Not implemented yet.")
    def test_call_expected_output(self):
        ...

    @pytest.mark.parametrize("input_shape",[
        tf.TensorShape([1]),
        tf.TensorShape([8,16]),
        tf.TensorShape([8,16,16]),
        tf.TensorShape([8,16,16,4]),
    ])
    def test_call_expected_shape(self,input_shape):
        layer = ScaleLayer()
        output = layer(tf.ones(shape=input_shape))
        assert output.shape == input_shape
