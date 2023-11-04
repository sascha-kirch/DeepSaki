import os

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
from abc import ABC
from abc import abstractmethod

import tensorflow as tf

from DeepSaki.models.autoencoders import ResNet
from DeepSaki.models.autoencoders import UNet
from DeepSaki.models.discriminators import LayoutContentDiscriminator
from DeepSaki.models.discriminators import PatchDiscriminator
from DeepSaki.models.discriminators import UNetDiscriminator
from tests.DeepSaki_test.layers_test.mocked_layers import _mock_bottleneck
from tests.DeepSaki_test.layers_test.mocked_layers import _mock_decoder
from tests.DeepSaki_test.layers_test.mocked_layers import _mock_encoder


class DeepSakiModelChecks(ABC):
    @abstractmethod
    def test_call_raises_error_wrong_input_spec(self):
        ...

    @abstractmethod
    def test_call_correct_output_shape(self):
        ...


class CommonModelChecks:
    @staticmethod
    def has_call_correct_output_shape(model_instance, input_shape, expected_shape, make_input_complex=False):
        input = tf.ones(shape=input_shape)
        if make_input_complex:
            input = tf.complex(real=input, imag=input)
        output = model_instance(input)
        assert output.shape == expected_shape

    @staticmethod
    def does_call_raises_error_wrong_input_spec(
        model_instance, input_shape, expected_context, make_input_complex=False
    ):
        with expected_context:
            input = tf.ones(shape=input_shape)
            if make_input_complex:
                input = tf.complex(real=input, imag=input)
            _ = model_instance(input)


@pytest.mark.parametrize(
    ("model_class", "calling_module"),
    [
        (UNet, "DeepSaki.models.autoencoders"),
        (ResNet, "DeepSaki.models.autoencoders"),
        (UNetDiscriminator, "DeepSaki.models.discriminators"),
        (PatchDiscriminator, "DeepSaki.models.discriminators"),
        (LayoutContentDiscriminator, "DeepSaki.models.discriminators"),
    ],
)
class TestGenericModel:
    @pytest.fixture(autouse=True)
    def mock_sub_models(self, mocker, calling_module):
        _mock_encoder(mocker, calling_module)
        _mock_bottleneck(mocker, calling_module)
        _mock_decoder(mocker, calling_module)

    def test_layer_is_subclass_of_tensorflow_model(self, model_class, **kwargs):
        assert isinstance(model_class(), tf.keras.Model)

    def test_input_spec_is_defined_in_init(self, model_class, **kwargs):
        assert model_class().input_spec is not None, "'tf.keras.layers.InputSpec' is not defined."
