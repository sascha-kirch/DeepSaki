import os
from contextlib import nullcontext as does_not_raise

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf

from DeepSaki.layers.layer_composites import Conv2DSplitted
from DeepSaki.layers.layer_composites import PaddingType
from DeepSaki.layers.layer_composites import ResidualBlock
from DeepSaki.layers.layer_composites import ScalarGatedSelfAttention
from DeepSaki.layers.layer_composites import ScaleLayer
from tests.DeepSaki_test.layers_test.layers_test import CommonLayerChecks
from tests.DeepSaki_test.layers_test.layers_test import DeepSakiLayerChecks


class TestConv2DSplitted(DeepSakiLayerChecks):
    @pytest.fixture()
    def conv_2d_splitted(self):
        return Conv2DSplitted()

    @pytest.mark.parametrize("use_spec_norm", [True, False])
    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize(
        ("input", "filters", "kernels", "strides", "padding", "expected_shape"),
        [
            (tf.TensorShape((1, 16, 16, 3)), 3, 3, (1, 1), "same", tf.TensorShape((1, 16, 16, 3))),
            (tf.TensorShape((8, 16, 16, 3)), 5, 5, (1, 1), "same", tf.TensorShape((8, 16, 16, 5))),
            (tf.TensorShape((1, 16, 16, 3)), 3, 3, (2, 2), "same", tf.TensorShape((1, 8, 8, 3))),
            (tf.TensorShape((1, 16, 16, 3)), 3, 5, (1, 2), "same", tf.TensorShape((1, 16, 8, 3))),
            (tf.TensorShape((1, 16, 16, 3)), 3, 3, (2, 1), "same", tf.TensorShape((1, 8, 16, 3))),
            (tf.TensorShape((1, 16, 16, 3)), 3, 5, (1, 1), "valid", tf.TensorShape((1, 12, 12, 3))),
            (tf.TensorShape((1, 16, 16, 3)), 5, 3, (1, 1), "valid", tf.TensorShape((1, 14, 14, 5))),
            (tf.TensorShape((1, 16, 16, 3)), 3, 3, (2, 2), "valid", tf.TensorShape((1, 7, 7, 3))),
        ],
    )
    def test_call_correct_output_shape(
        self, input, filters, strides, padding, use_bias, use_spec_norm, kernels, expected_shape
    ):
        layer_instance = Conv2DSplitted(filters, kernels, use_spec_norm, strides, use_bias, padding)
        CommonLayerChecks.has_call_correct_output_shape(layer_instance, input, expected_shape)

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
    def test_call_raises_error_wrong_input_spec(self, conv_2d_splitted, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(conv_2d_splitted, input_shape, expected_context)


class TestResidualBlock(DeepSakiLayerChecks):
    @pytest.fixture()
    def residual_block(self):
        return ResidualBlock()

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
    @pytest.mark.skip(reason="Not implemented yet.")
    def test_call_raises_error_wrong_input_spec(self, residual_block, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(residual_block, input_shape, expected_context)

    @pytest.mark.parametrize("use_spec_norm", [False])
    @pytest.mark.parametrize("use_bias", [True])
    @pytest.mark.parametrize("activation", ["leaky_relu"])
    @pytest.mark.parametrize("kernels", [3])
    @pytest.mark.parametrize("dropout_rate", [0.0, 0.5])
    @pytest.mark.parametrize("number_of_blocks", [1, 2])
    @pytest.mark.parametrize("residual_cardinality", [1, 3])
    @pytest.mark.parametrize("padding", [PaddingType.ZERO])
    @pytest.mark.parametrize(
        ("input", "filters", "expected_shape"),
        [
            (tf.TensorShape((1, 16, 16, 3)), 3, tf.TensorShape((1, 16, 16, 3))),
            (tf.TensorShape((1, 16, 16, 3)), 8, tf.TensorShape((1, 16, 16, 8))),
            (tf.TensorShape((1, 16, 16, 3)), 5, tf.TensorShape((1, 16, 16, 5))),
            (tf.TensorShape((4, 8, 8, 4)), 12, tf.TensorShape((4, 8, 8, 12))),
        ],
    )
    def test_call_correct_output_shape(
        self,
        input,
        expected_shape,
        filters,
        kernels,
        activation,
        number_of_blocks,
        use_spec_norm,
        residual_cardinality,
        dropout_rate,
        use_bias,
        padding,
    ):
        layer_instance = ResidualBlock(
            filters,
            kernels,
            activation,
            number_of_blocks,
            use_spec_norm,
            residual_cardinality,
            dropout_rate,
            use_bias,
            padding,
        )
        CommonLayerChecks.has_call_correct_output_shape(layer_instance, input, expected_shape)

    @pytest.mark.parametrize("number_of_blocks", [1, 2, 3])
    def test_number_of_blocks_correct(self, number_of_blocks):
        layer = ResidualBlock(number_of_blocks=number_of_blocks)
        assert len(layer.blocks) == number_of_blocks

    @pytest.mark.parametrize("residual_cardinality", [1, 2, 3])
    @pytest.mark.parametrize("number_of_blocks", [1, 2, 3])
    def test_residual_cardinality_correct(self, number_of_blocks, residual_cardinality):
        layer = ResidualBlock(number_of_blocks=number_of_blocks, residual_cardinality=residual_cardinality)
        for block in layer.blocks:
            assert len(block) == residual_cardinality


class TestScaleLayer(DeepSakiLayerChecks):
    @pytest.fixture()
    def scale_layer(self):
        return ScaleLayer()

    @pytest.mark.skip(reason="Not implemented yet.")
    def test_init(self):
        ...

    @pytest.mark.skip(reason="Not implemented yet.")
    def test_call_expected_output(self):
        ...

    @pytest.mark.parametrize(
        "input_shape",
        [
            tf.TensorShape([1]),
            tf.TensorShape([8, 16]),
            tf.TensorShape([8, 16, 16]),
            tf.TensorShape([8, 16, 16, 4]),
        ],
    )
    def test_call_correct_output_shape(self, scale_layer, input_shape):
        CommonLayerChecks.has_call_correct_output_shape(scale_layer, input_shape, input_shape)

    @pytest.mark.parametrize(
        ("input_shape", "expected_context"),
        [
            (tf.TensorShape((8)), does_not_raise()),
            (tf.TensorShape((8, 16)), does_not_raise()),
            (tf.TensorShape((8, 16, 16)), does_not_raise()),
            (tf.TensorShape((8, 16, 16, 4)), does_not_raise()),
            (tf.TensorShape((8, 16, 16, 4, 12)), does_not_raise()),
            (tf.TensorShape(()), pytest.raises(ValueError)),
        ],
    )
    def test_call_raises_error_wrong_input_spec(self, scale_layer, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(scale_layer, input_shape, expected_context)


class TestScalarGatedSelfAttention(DeepSakiLayerChecks):
    @pytest.fixture()
    def scalar_gated_self_attention(self):
        return ScalarGatedSelfAttention()

    @pytest.mark.parametrize("use_spec_norm", [True, False])
    @pytest.mark.parametrize("intermediate_channel", [3, 7, 12, None])
    @pytest.mark.parametrize(
        "input_shape",
        [
            tf.TensorShape((1, 8, 8, 3)),
            tf.TensorShape((8, 8, 8, 4)),
            tf.TensorShape((1, 3, 3, 1)),
        ],
    )
    def test_call_correct_output_shape(self, input_shape, intermediate_channel, use_spec_norm):
        layer_instance = ScalarGatedSelfAttention(use_spec_norm, intermediate_channel)
        CommonLayerChecks.has_call_correct_output_shape(layer_instance, input_shape, input_shape)

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
    def test_call_raises_error_wrong_input_spec(self, scalar_gated_self_attention, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(
            scalar_gated_self_attention, input_shape, expected_context
        )
