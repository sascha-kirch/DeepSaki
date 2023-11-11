import os
from contextlib import nullcontext as does_not_raise

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf

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
from DeepSaki.types.layers_enums import DownSampleType
from DeepSaki.types.layers_enums import PaddingType
from DeepSaki.types.layers_enums import UpSampleType
from tests.DeepSaki_test.layers_test.layers_test import CommonLayerChecks
from tests.DeepSaki_test.layers_test.layers_test import DeepSakiLayerChecks
from tests.DeepSaki_test.layers_test.mocked_layers import _mock_scale_layer

# TODO: ensure I only test inputs relevant for that particular layer and that are relevant for the test. esp. when testing the output shape.
# TODO: change expected output shape to be an equation. In that way it is easier to understand later and it is always the same shape, which is important for layers using this layer.


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


class TestConv2DBlock(DeepSakiLayerChecks):
    @pytest.fixture()
    def conv_2d_block(self):
        return Conv2DBlock()

    @pytest.mark.parametrize("use_spec_norm", [True, False])
    @pytest.mark.parametrize("split_kernels", [True, False])
    @pytest.mark.parametrize("use_bias", [False])
    @pytest.mark.parametrize("final_activation", [True, False])
    @pytest.mark.parametrize("apply_final_normalization", [True, False])
    @pytest.mark.parametrize("number_of_blocks", [1, 2])
    @pytest.mark.parametrize(
        ("input_shape", "filters", "kernels", "strides"),
        [
            (tf.TensorShape((1, 16, 16, 3)), 3, 3, (1, 1)),
            (tf.TensorShape((8, 16, 16, 3)), 5, 5, (1, 1)),
            (tf.TensorShape((1, 16, 16, 3)), 3, 3, (2, 2)),
            (tf.TensorShape((8, 16, 16, 3)), 5, 5, (1, 2)),
            (tf.TensorShape((1, 16, 16, 3)), 1, 3, (2, 1)),
        ],
    )
    def test_call_correct_output_shape(
        self,
        input_shape,
        filters,
        strides,
        use_bias,
        use_spec_norm,
        kernels,
        number_of_blocks,
        split_kernels,
        final_activation,
        apply_final_normalization,
    ):
        layer_instance = Conv2DBlock(
            filters,
            kernels,
            split_kernels,
            number_of_blocks,
            final_activation=final_activation,
            apply_final_normalization=apply_final_normalization,
            use_spec_norm=use_spec_norm,
            strides=strides,
            use_bias=use_bias,
        )
        expected_shape = (
            input_shape[0],
            input_shape[1] // (strides[0] ** number_of_blocks),
            input_shape[2] // (strides[1] ** number_of_blocks),
            filters,
        )
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
    def test_call_raises_error_wrong_input_spec(self, conv_2d_block, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(conv_2d_block, input_shape, expected_context)

    @pytest.mark.parametrize("number_of_blocks", [1, 2, 3])
    def test_number_of_blocks_correct(self, number_of_blocks):
        layer_instance = Conv2DBlock(number_of_blocks=number_of_blocks)
        assert len(layer_instance.blocks) == number_of_blocks

    @pytest.mark.parametrize("number_of_blocks", [1, 2, 3])
    @pytest.mark.parametrize(
        ("apply_final_normalization", "final_activation", "expected_final_len"),
        [
            (True, True, 4),
            (True, False, 3),
            (False, True, 3),
            (False, False, 2),
        ],
    )
    def test_final_normalization_activation_correctly_set(
        self, number_of_blocks, apply_final_normalization, final_activation, expected_final_len
    ):
        layer_instance = Conv2DBlock(
            number_of_blocks=number_of_blocks,
            apply_final_normalization=apply_final_normalization,
            final_activation=final_activation,
        )
        assert len(layer_instance.blocks[-1]) == expected_final_len


class TestDenseBlock(DeepSakiLayerChecks):
    @pytest.fixture()
    def dense_block(self):
        return DenseBlock(16)

    @pytest.mark.parametrize("use_spec_norm", [True, False])
    @pytest.mark.parametrize("use_bias", [False])
    @pytest.mark.parametrize("final_activation", [True, False])
    @pytest.mark.parametrize("apply_final_normalization", [True, False])
    @pytest.mark.parametrize("number_of_blocks", [1, 2])
    @pytest.mark.parametrize("units", [8, 16, 32])
    @pytest.mark.parametrize(
        ("input_shape"),
        [
            (tf.TensorShape((1, 16, 16, 3))),
            (tf.TensorShape((8, 32, 20))),
            (tf.TensorShape((8, 32))),
        ],
    )
    def test_call_correct_output_shape(
        self, input_shape, units, use_bias, use_spec_norm, number_of_blocks, final_activation, apply_final_normalization
    ):
        layer_instance = DenseBlock(
            units,
            number_of_blocks,
            final_activation=final_activation,
            apply_final_normalization=apply_final_normalization,
            use_spec_norm=use_spec_norm,
            use_bias=use_bias,
        )
        expected_shape = [*input_shape[0:-1], units]
        CommonLayerChecks.has_call_correct_output_shape(layer_instance, input_shape, expected_shape)

    @pytest.mark.parametrize(
        ("input_shape", "expected_context"),
        [
            (tf.TensorShape((8, 64, 64, 4)), does_not_raise()),
            (tf.TensorShape((1, 32, 32, 3)), does_not_raise()),
            (tf.TensorShape((8, 64, 64)), does_not_raise()),
            (tf.TensorShape((8, 64)), does_not_raise()),
            (tf.TensorShape((8)), pytest.raises(ValueError)),
            (tf.TensorShape((8, 64, 64, 4, 5)), does_not_raise()),
            (tf.TensorShape((8, 64, 64, 4, 5, 6)), does_not_raise()),
        ],
    )
    def test_call_raises_error_wrong_input_spec(self, dense_block, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(dense_block, input_shape, expected_context)

    @pytest.mark.parametrize("number_of_blocks", [1, 2, 3])
    def test_number_of_blocks_correct(self, number_of_blocks):
        layer_instance = DenseBlock(units=16, number_of_blocks=number_of_blocks)
        assert len(layer_instance.blocks) == number_of_blocks

    @pytest.mark.parametrize("number_of_blocks", [1, 2, 3])
    @pytest.mark.parametrize(
        ("apply_final_normalization", "final_activation", "expected_final_len"),
        [
            (True, True, 3),
            (True, False, 2),
            (False, True, 2),
            (False, False, 1),
        ],
    )
    def test_final_normalization_activation_correctly_set(
        self, number_of_blocks, apply_final_normalization, final_activation, expected_final_len
    ):
        layer_instance = DenseBlock(
            units=16,
            number_of_blocks=number_of_blocks,
            apply_final_normalization=apply_final_normalization,
            final_activation=final_activation,
        )
        assert len(layer_instance.blocks[-1]) == expected_final_len


class TestDownSamplingBlock(DeepSakiLayerChecks):
    @pytest.fixture()
    def downsample_block(self):
        return DownSampleBlock()

    @pytest.mark.parametrize(
        "downsampling",
        [
            DownSampleType.AVG_POOLING,
            DownSampleType.MAX_POOLING,
            DownSampleType.CONV_STRIDE_2,
            DownSampleType.SPACE_TO_DEPTH,
        ],
    )
    @pytest.mark.parametrize("kernels", [3, 5])
    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize(
        ("input_shape"),
        [
            tf.TensorShape((1, 16, 16, 4)),
        ],
    )
    def test_call_correct_output_shape(self, input_shape, downsampling, kernels, use_bias):
        layer_instance = DownSampleBlock(downsampling=downsampling, kernels=kernels, use_bias=use_bias)
        expected_shape = [
            input_shape[0],
            input_shape[1] // 2,
            input_shape[2] // 2,
            input_shape[3],
        ]
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
    def test_call_raises_error_wrong_input_spec(self, downsample_block, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(downsample_block, input_shape, expected_context)

    @pytest.mark.parametrize(
        ("downsampling", "expected_context"),
        [
            (DownSampleType.CONV_STRIDE_2, does_not_raise()),
            (DownSampleType.MAX_POOLING, does_not_raise()),
            (DownSampleType.AVG_POOLING, does_not_raise()),
            (DownSampleType.SPACE_TO_DEPTH, does_not_raise()),
            ("Any other String", pytest.raises(ValueError)),
        ],
    )
    def test_build_raises_error_wrong_downsampling(self, downsampling, expected_context):
        layer_instance = DownSampleBlock(downsampling=downsampling)
        with expected_context:
            layer_instance.build(input_shape=(1, 16, 16, 4))


class TestUpSamplingBlock(DeepSakiLayerChecks):
    @pytest.fixture()
    def upsample_block(self):
        return UpSampleBlock()

    @pytest.mark.parametrize(
        "upsampling", [UpSampleType.RESAMPLE_AND_CONV, UpSampleType.TRANSPOSE_CONV, UpSampleType.DEPTH_TO_SPACE]
    )
    @pytest.mark.parametrize("kernels", [3, 5])
    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize(
        ("input_shape"),
        [
            tf.TensorShape((1, 16, 16, 4)),
        ],
    )
    def test_call_correct_output_shape(self, input_shape, upsampling, kernels, use_bias):
        layer_instance = UpSampleBlock(upsampling=upsampling, kernels=kernels, use_bias=use_bias)
        expected_shape = [
            input_shape[0],
            input_shape[1] * 2,
            input_shape[2] * 2,
            input_shape[3],
        ]
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
    def test_call_raises_error_wrong_input_spec(self, upsample_block, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(upsample_block, input_shape, expected_context)

    @pytest.mark.parametrize(
        ("upsampling", "expected_context"),
        [
            (UpSampleType.RESAMPLE_AND_CONV, does_not_raise()),
            (UpSampleType.TRANSPOSE_CONV, does_not_raise()),
            (UpSampleType.DEPTH_TO_SPACE, does_not_raise()),
            ("Any other String", pytest.raises(ValueError)),
        ],
    )
    def test_build_raises_error_wrong_downsampling(self, upsampling, expected_context):
        layer_instance = UpSampleBlock(upsampling=upsampling)
        with expected_context:
            layer_instance.build(input_shape=(1, 16, 16, 4))


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


class TestResBlockDown(DeepSakiLayerChecks):
    @pytest.fixture()
    def resblock_down(self):
        return ResBlockDown()

    @pytest.mark.parametrize(
        ("input_shape"),
        [
            tf.TensorShape((1, 16, 16, 4)),
            tf.TensorShape((8, 32, 32, 3)),
            tf.TensorShape((5, 32, 16, 12)),
            tf.TensorShape((3, 16, 32, 8)),
        ],
    )
    def test_call_correct_output_shape(self, input_shape):
        layer_instance = ResBlockDown()
        expected_shape = [
            input_shape[0],
            input_shape[1] // 2,
            input_shape[2] // 2,
            input_shape[3],
        ]
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
    def test_call_raises_error_wrong_input_spec(self, resblock_down, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(resblock_down, input_shape, expected_context)


class TestResBlockUp(DeepSakiLayerChecks):
    @pytest.fixture()
    def resblock_up(self):
        return ResBlockUp()

    @pytest.mark.parametrize(
        ("input_shape"),
        [
            tf.TensorShape((1, 16, 16, 4)),
            tf.TensorShape((8, 32, 32, 3)),
            tf.TensorShape((5, 32, 16, 12)),
            tf.TensorShape((3, 16, 32, 8)),
        ],
    )
    def test_call_correct_output_shape(self, input_shape):
        layer_instance = ResBlockUp()
        expected_shape = [
            input_shape[0],
            input_shape[1] * 2,
            input_shape[2] * 2,
            input_shape[3],
        ]
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
    def test_call_raises_error_wrong_input_spec(self, resblock_up, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(resblock_up, input_shape, expected_context)


class TestScaleLayer(DeepSakiLayerChecks):
    @pytest.fixture()
    def scale_layer(self):
        return ScaleLayer()

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

    @pytest.fixture(autouse=True)
    def mock_layers(self, mocker):
        calling_module = "DeepSaki.layers.layer_composites"
        # _mock_conv2d_block(mocker,calling_module) mock is not yet implemented
        _mock_scale_layer(mocker, calling_module)

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
