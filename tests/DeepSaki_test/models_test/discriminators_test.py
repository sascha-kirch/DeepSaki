import os
from contextlib import nullcontext as does_not_raise

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf

from DeepSaki.models.discriminators import LayoutContentDiscriminator
from DeepSaki.models.discriminators import PatchDiscriminator
from DeepSaki.models.discriminators import UNetDiscriminator
from tests.DeepSaki_test.layers_test.mocked_layers import _mock_bottleneck
from tests.DeepSaki_test.layers_test.mocked_layers import _mock_decoder
from tests.DeepSaki_test.layers_test.mocked_layers import _mock_encoder
from tests.DeepSaki_test.layers_test.mocked_layers import _mock_global_sum_pooling_2d
from tests.DeepSaki_test.layers_test.mocked_layers import _mock_scalar_gated_self_attention
from tests.DeepSaki_test.models_test.models_test import CommonModelChecks
from tests.DeepSaki_test.models_test.models_test import DeepSakiModelChecks
from DeepSaki.types.layers_enums import LinearLayerType


class TestLayoutContentDiscriminator(DeepSakiModelChecks):
    @pytest.fixture()
    def model_instance(self):
        return LayoutContentDiscriminator(filters=64, number_of_blocks=2)

    @pytest.fixture(autouse=True)
    def mock_sub_models(self, mocker):
        calling_module = "DeepSaki.models.discriminators"
        _mock_encoder(mocker, calling_module)
        _mock_scalar_gated_self_attention(mocker, calling_module)

    @pytest.mark.parametrize(
        ("input_shape", "expected_context"),
        [
            (tf.TensorShape((1, 256, 256, 4)), does_not_raise()),
            (tf.TensorShape((1, 256, 256, 3)), does_not_raise()),
            (tf.TensorShape((8, 64, 64)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 64)), pytest.raises(ValueError)),
            (tf.TensorShape((8)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5, 6)), pytest.raises(ValueError)),
        ],
    )
    def test_call_raises_error_wrong_input_spec(self, model_instance, input_shape, expected_context):
        CommonModelChecks.does_call_raises_error_wrong_input_spec(model_instance, input_shape, expected_context)

    @pytest.mark.parametrize(
        ("input_shape", "expected_context"),
        [
            (tf.TensorShape((8, 256, 256, 4)), does_not_raise()),
            (tf.TensorShape((1, 344, 344, 3)), does_not_raise()),
            (tf.TensorShape((8, 64, 64, 3)), pytest.raises(ValueError)),
            (tf.TensorShape((1, 256, 64, 3)), pytest.raises(ValueError)),
            (tf.TensorShape((1, 64, 256, 3)), pytest.raises(ValueError)),
        ],
    )
    def test_build_raises_error_wrong_input_height_width(self, model_instance, input_shape, expected_context):
        with expected_context:
            model_instance.build(input_shape)

    @pytest.mark.parametrize("linear_layer_type", [LinearLayerType.MLP, LinearLayerType.CONV_1x1])
    @pytest.mark.parametrize("use_self_attention", [True, False])
    @pytest.mark.parametrize("filters", [8, 16])
    @pytest.mark.parametrize(
        ("input_shape"),
        [
            tf.TensorShape((1, 512, 512, 3)),
            tf.TensorShape((8, 256, 256, 4)),
        ],
    )
    def test_call_correct_output_shape(self, input_shape, use_self_attention, filters, linear_layer_type):
        model_instance = LayoutContentDiscriminator(
            use_self_attention=use_self_attention, filters=filters, linear_layer_type=linear_layer_type
        )
        content_output, layout_output = model_instance(tf.ones(shape=input_shape))
        expected_shape_layout_output = [
            input_shape[0],
            input_shape[1] // 2**3,  # 3 is the number of downsampling blocks of the encoder
            input_shape[2] // 2**3,
            1,
        ]
        expected_shape_content_output = [
            input_shape[0],
            1,
            1,
            filters * 2**3,  # 3 is the number of downsampling blocks of the encoder
        ]

        assert layout_output.shape == expected_shape_layout_output
        assert content_output.shape == expected_shape_content_output


class TestPatchDiscriminator(DeepSakiModelChecks):
    @pytest.fixture()
    def model_instance(self):
        return PatchDiscriminator(filters=8, number_of_blocks=1)

    @pytest.fixture(autouse=True)
    def mock_sub_models(self, mocker):
        calling_module = "DeepSaki.models.discriminators"
        _mock_encoder(mocker, calling_module)

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

    @pytest.mark.parametrize("num_down_blocks", [2, 3, 4])
    @pytest.mark.parametrize("filters", [8, 16])
    @pytest.mark.parametrize(
        ("input_shape"),
        [
            tf.TensorShape((1, 64, 64, 3)),
            tf.TensorShape((8, 64, 64, 4)),
        ],
    )
    def test_call_correct_output_shape(self, input_shape, num_down_blocks, filters):
        model_instance = PatchDiscriminator(num_down_blocks=num_down_blocks, filters=filters)
        expected_shape = [
            input_shape[0],
            input_shape[1] // 2**num_down_blocks,
            input_shape[2] // 2**num_down_blocks,
            1,
        ]
        CommonModelChecks.has_call_correct_output_shape(model_instance, input_shape, expected_shape)


class TestUNetDiscriminator(DeepSakiModelChecks):
    @pytest.fixture()
    def model_instance(self):
        return UNetDiscriminator(number_of_levels=2, filters=8, number_of_blocks=1)

    @pytest.fixture(autouse=True)
    def mock_sub_models(self, mocker):
        calling_module = "DeepSaki.models.discriminators"
        _mock_encoder(mocker, calling_module)
        _mock_bottleneck(mocker, calling_module)
        _mock_decoder(mocker, calling_module)
        _mock_global_sum_pooling_2d(mocker, calling_module)

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
    def test_call_correct_output_shape(self, input_shape, number_of_levels, filters, linear_layer_type):
        model_instance = UNetDiscriminator(
            number_of_levels=number_of_levels, filters=filters, linear_layer_type=linear_layer_type
        )
        global_output, decoder_output = model_instance(tf.ones(shape=input_shape))
        expected_shape_decoder_output = [*input_shape[0:-1], 1]
        expected_shape_global_output = [input_shape[0], 1]

        assert decoder_output.shape == expected_shape_decoder_output
        assert global_output.shape == expected_shape_global_output
