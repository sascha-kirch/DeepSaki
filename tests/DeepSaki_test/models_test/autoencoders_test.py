import os
from contextlib import nullcontext as does_not_raise

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf

from DeepSaki.models.autoencoders import ResNet
from DeepSaki.models.autoencoders import UNet
from tests.DeepSaki_test.layers_test.mocked_layers import _mock_bottleneck
from tests.DeepSaki_test.layers_test.mocked_layers import _mock_decoder
from tests.DeepSaki_test.layers_test.mocked_layers import _mock_encoder
from tests.DeepSaki_test.models_test.models_test import CommonModelChecks
from tests.DeepSaki_test.models_test.models_test import DeepSakiModelChecks
from DeepSaki.types.layers_enums import LinearLayerType


@pytest.mark.parametrize(("autoencoder_model"), [UNet, ResNet])
class TestGenericAutoEncoder(DeepSakiModelChecks):
    @pytest.fixture(autouse=True)
    def mock_sub_models(self, mocker):
        calling_module = "DeepSaki.models.autoencoders"
        _mock_encoder(mocker, calling_module)
        _mock_bottleneck(mocker, calling_module)
        _mock_decoder(mocker, calling_module)

    @pytest.fixture()
    def model_instance(self, autoencoder_model):
        return autoencoder_model(number_of_levels=2, filters=8, number_of_blocks=1)

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
    def test_call_raises_error_wrong_input_spec(self, model_instance, input_shape, expected_context):
        CommonModelChecks.does_call_raises_error_wrong_input_spec(model_instance, input_shape, expected_context)

    @pytest.mark.parametrize("linear_layer_type", [LinearLayerType.MLP, LinearLayerType.CONV_1x1])
    @pytest.mark.parametrize("number_of_levels", [2, 3, 4])
    @pytest.mark.parametrize("filters", [8, 16])
    @pytest.mark.parametrize(
        ("input_shape"),
        [
            tf.TensorShape((1, 64, 64, 3)),
            tf.TensorShape((8, 64, 64, 4)),
        ],
    )
    def test_call_correct_output_shape(
        self, autoencoder_model, input_shape, number_of_levels, filters, linear_layer_type
    ):
        model_instance = autoencoder_model(
            number_of_levels=number_of_levels, filters=filters, linear_layer_type=linear_layer_type
        )
        CommonModelChecks.has_call_correct_output_shape(model_instance, input_shape, input_shape)

    @pytest.mark.parametrize(
        ("linear_layer_type", "expected_context"),
        [
            (LinearLayerType.MLP, does_not_raise()),
            (LinearLayerType.CONV_1x1, does_not_raise()),
            ("Any other String", pytest.raises(ValueError)),
        ],
    )
    def test_init_raises_wrong_linear_layer_type_type(self, autoencoder_model, linear_layer_type, expected_context):
        with expected_context:
            _ = autoencoder_model(number_of_levels=2, filters=8, linear_layer_type=linear_layer_type)
