import os

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
from contextlib import nullcontext as does_not_raise

import tensorflow as tf

from DeepSaki.layers.sub_model_composites import Bottleneck
from DeepSaki.layers.sub_model_composites import Decoder
from DeepSaki.layers.sub_model_composites import Encoder
from tests.DeepSaki_test.layers_test.layers_test import CommonLayerChecks
from tests.DeepSaki_test.layers_test.layers_test import DeepSakiLayerChecks
from tests.DeepSaki_test.layers_test.mocked_layers import _mock_downsample_block
from tests.DeepSaki_test.layers_test.mocked_layers import _mock_resblock_down
from tests.DeepSaki_test.layers_test.mocked_layers import _mock_resblock_up
from tests.DeepSaki_test.layers_test.mocked_layers import _mock_residualblock
from tests.DeepSaki_test.layers_test.mocked_layers import _mock_scalar_gated_self_attention
from tests.DeepSaki_test.layers_test.mocked_layers import _mock_upsample_block


class TestEncoder(DeepSakiLayerChecks):
    @pytest.fixture()
    def encoder(self):
        return Encoder(number_of_levels=2, filters=8, number_of_blocks=2)

    @pytest.fixture(autouse=True)
    def mock_layers(self, mocker):
        calling_module = "DeepSaki.layers.sub_model_composites"
        # _mock_conv2d_block(mocker,calling_module) mock is not yet implemented
        _mock_residualblock(mocker, calling_module)
        _mock_resblock_down(mocker, calling_module)
        _mock_downsample_block(mocker, calling_module)
        _mock_scalar_gated_self_attention(mocker, calling_module)

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
    def test_call_raises_error_wrong_input_spec(self, encoder, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(encoder, input_shape, expected_context)

    @pytest.mark.parametrize("use_ResidualBlock", [True, False])
    @pytest.mark.parametrize("use_self_attention", [True, False])
    @pytest.mark.parametrize("number_of_levels", [3, 4])
    @pytest.mark.parametrize("filters", [2, 4])
    @pytest.mark.parametrize(
        "input_shape",
        [
            tf.TensorShape((1, 16, 16, 3)),
            tf.TensorShape((8, 32, 32, 4)),
        ],
    )
    def test_call_correct_output_shape(
        self, input_shape, number_of_levels, filters, use_ResidualBlock, use_self_attention
    ):
        layer_instance = Encoder(
            number_of_levels, filters, use_ResidualBlock=use_ResidualBlock, use_self_attention=use_self_attention
        )
        expected_shape = (
            input_shape[0],
            input_shape[1] // 2**number_of_levels,
            input_shape[2] // 2**number_of_levels,
            filters * 2 ** (number_of_levels - 1),
        )
        CommonLayerChecks.has_call_correct_output_shape(layer_instance, input_shape, expected_shape)

    @pytest.mark.parametrize(
        ("number_of_levels", "channel_list", "expected_levels"),
        [
            (1, None, 1),
            (3, None, 3),
            (5, None, 5),
            (7, None, 7),
            (1, (8, 16), 2),
            (3, (8, 8, 8), 3),
            (5, (4, 4, 4, 4), 4),
            (7, (4, 4, 4, 4, 4), 5),
        ],
    )
    def test_number_of_levels_correct(self, number_of_levels, channel_list, expected_levels):
        layer_instance = Encoder(
            number_of_levels=number_of_levels, channel_list=channel_list, filters=4, number_of_blocks=1
        )
        layer_instance.build(tf.TensorShape((1, 16, 16, 4)))
        assert len(layer_instance.encoderBlocks) == len(layer_instance.downSampleBlocks) == expected_levels

    @pytest.mark.parametrize("use_self_attention", [(True, False)])
    def test_self_attention_is_not_none(self, use_self_attention):
        layer_instance = Encoder(
            use_self_attention=use_self_attention, number_of_levels=1, filters=4, number_of_blocks=1
        )
        layer_instance.build(tf.TensorShape((1, 16, 16, 4)))
        if use_self_attention:
            assert layer_instance.SA is not None
        else:
            assert layer_instance.SA is None

    @pytest.mark.parametrize(
        ("number_of_levels", "channel_list", "filters", "limit_filters", "expected_channel_list"),
        [
            (1, None, 3, 128, [3]),
            (3, None, 8, 128, [8, 16, 32]),
            (6, None, 8, 128, [8, 16, 32, 64, 128, 128]),
            (6, [8, 16, 32, 256], 8, 128, [8, 16, 32, 256]),
        ],
    )
    def test_channel_list_as_expected(
        self, number_of_levels, channel_list, filters, limit_filters, expected_channel_list
    ):
        layer_instance = Encoder(
            number_of_levels=number_of_levels,
            limit_filters=limit_filters,
            channel_list=channel_list,
            filters=filters,
            number_of_blocks=1,
        )
        layer_instance.build(tf.TensorShape((1, 16, 16, 4)))
        assert layer_instance.channel_list == expected_channel_list

    @pytest.mark.parametrize(
        ("output_skips", "omit_skips", "number_of_levels", "expected_skips", "expected_context"),
        [
            (True, 0, 3, [True, True, True], does_not_raise()),
            (True, 1, 3, [False, True, True], does_not_raise()),
            (True, 3, 3, [False, False, False], does_not_raise()),
            (False, 0, 3, [False, False, False], pytest.raises(ValueError, match="(expected 2, got 1)")),
        ],
    )
    def test_correct_skips_output(self, output_skips, omit_skips, expected_skips, number_of_levels, expected_context):
        layer_instance = Encoder(
            output_skips=output_skips, omit_skips=omit_skips, number_of_levels=number_of_levels, number_of_blocks=1
        )
        with expected_context:
            _, skips = layer_instance(tf.ones(shape=(1, 16, 16, 4)))
            assert len(skips) == len(expected_skips)
            for skip, expected_skip in zip(skips, expected_skips):
                assert tf.is_tensor(skip) == expected_skip

    @pytest.mark.parametrize(
        ("input_shape", "channel_list", "omit_skips", "expected_shapes"),
        [
            ((1, 16, 16, 3), [4, 8, 16], 0, [(1, 16, 16, 4), (1, 8, 8, 8), (1, 4, 4, 16)]),
            ((1, 16, 16, 3), [4, 8, 16], 1, [None, (1, 8, 8, 8), (1, 4, 4, 16)]),
            ((1, 16, 16, 3), [4, 8, 16], 2, [None, None, (1, 4, 4, 16)]),
        ],
    )
    def test_skips_output_expected_shape(self, input_shape, channel_list, omit_skips, expected_shapes):
        layer_instance = Encoder(
            output_skips=True, omit_skips=omit_skips, channel_list=channel_list, number_of_blocks=1
        )
        _, skips = layer_instance(tf.ones(shape=input_shape))
        for skip, expected_shape in zip(skips, expected_shapes):
            if not tf.is_tensor(skip):  # if skip is ommited for current level
                continue
            assert skip.shape == expected_shape


class TestBottleneck(DeepSakiLayerChecks):
    @pytest.fixture()
    def bottleneck(self):
        return Bottleneck(n_bottleneck_blocks=1, number_of_blocks=1)

    @pytest.fixture(autouse=True)
    def mock_layers(self, mocker):
        calling_module = "DeepSaki.layers.sub_model_composites"
        # _mock_conv2d_block(mocker,calling_module) mock is not yet implemented
        _mock_residualblock(mocker, calling_module)

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
    def test_call_raises_error_wrong_input_spec(self, bottleneck, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(bottleneck, input_shape, expected_context)

    @pytest.mark.parametrize("use_ResidualBlock", [True, False])
    @pytest.mark.parametrize("n_bottleneck_blocks", [3, 4])
    @pytest.mark.parametrize(
        "input_shape",
        [
            tf.TensorShape((1, 16, 16, 3)),
            tf.TensorShape((8, 32, 32, 4)),
        ],
    )
    def test_call_correct_output_shape(
        self,
        input_shape,
        n_bottleneck_blocks,
        use_ResidualBlock,
    ):
        layer_instance = Bottleneck(n_bottleneck_blocks, use_ResidualBlock=use_ResidualBlock)
        expected_shape = input_shape
        CommonLayerChecks.has_call_correct_output_shape(layer_instance, input_shape, expected_shape)

    @pytest.mark.parametrize(
        ("n_bottleneck_blocks", "channel_list", "expected_levels"),
        [
            (1, None, 1),
            (3, None, 3),
            (5, None, 5),
            (7, None, 7),
            (1, (8, 16), 2),
            (3, (8, 8, 8), 3),
            (5, (4, 4, 4, 4), 4),
            (7, (4, 4, 4, 4, 4), 5),
        ],
    )
    def test_number_of_levels_correct(self, n_bottleneck_blocks, channel_list, expected_levels):
        layer_instance = Bottleneck(
            n_bottleneck_blocks=n_bottleneck_blocks, channel_list=channel_list, number_of_blocks=1
        )
        layer_instance.build(tf.TensorShape((1, 16, 16, 4)))
        assert len(layer_instance.layers) == expected_levels

    @pytest.mark.parametrize(
        ("input_shape", "n_bottleneck_blocks", "channel_list", "expected_channel_list"),
        [
            (tf.TensorShape((1, 16, 16, 4)), 1, None, [4]),
            (tf.TensorShape((1, 16, 16, 4)), 3, None, [4, 4, 4]),
            (tf.TensorShape((1, 16, 16, 4)), 6, [8, 10, 12, 14], [8, 10, 12, 14]),
        ],
    )
    def test_channel_list_as_expected(self, input_shape, n_bottleneck_blocks, channel_list, expected_channel_list):
        layer_instance = Bottleneck(
            n_bottleneck_blocks=n_bottleneck_blocks, channel_list=channel_list, number_of_blocks=1
        )
        layer_instance.build(input_shape)
        assert layer_instance.channel_list == expected_channel_list


class TestDecoder(DeepSakiLayerChecks):
    @pytest.fixture()
    def decoder(self):
        return Decoder(number_of_levels=2, filters=8, number_of_blocks=2)

    @pytest.fixture(autouse=True)
    def mock_layers(self, mocker):
        calling_module = "DeepSaki.layers.sub_model_composites"
        # _mock_conv2d_block(mocker,calling_module) mock is not yet implemented
        _mock_residualblock(mocker, calling_module)
        _mock_resblock_up(mocker, calling_module)
        _mock_upsample_block(mocker, calling_module)
        _mock_scalar_gated_self_attention(mocker, calling_module)

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
    def test_call_raises_error_wrong_input_spec(self, decoder, input_shape, expected_context):
        CommonLayerChecks.does_call_raises_error_wrong_input_spec(decoder, input_shape, expected_context)

    @pytest.mark.parametrize("use_ResidualBlock", [True, False])
    @pytest.mark.parametrize("use_self_attention", [True, False])
    @pytest.mark.parametrize("number_of_levels", [3, 4])
    @pytest.mark.parametrize("filters", [2, 4])
    @pytest.mark.parametrize(
        "input_shape",
        [
            tf.TensorShape((1, 2, 2, 32)),
            tf.TensorShape((8, 4, 4, 64)),
        ],
    )
    def test_call_correct_output_shape(
        self, input_shape, number_of_levels, filters, use_ResidualBlock, use_self_attention
    ):
        layer_instance = Decoder(
            number_of_levels=number_of_levels,
            filters=filters,
            use_ResidualBlock=use_ResidualBlock,
            use_self_attention=use_self_attention,
        )
        expected_shape = (
            input_shape[0],
            input_shape[1] * 2**number_of_levels,
            input_shape[2] * 2**number_of_levels,
            filters,
        )
        CommonLayerChecks.has_call_correct_output_shape(layer_instance, input_shape, expected_shape)

    @pytest.mark.parametrize(
        ("number_of_levels", "channel_list", "expected_levels"),
        [
            (1, None, 1),
            (3, None, 3),
            (5, None, 5),
            (7, None, 7),
            (1, (8, 16), 2),
            (3, (8, 8, 8), 3),
            (5, (4, 4, 4, 4), 4),
            (7, (4, 4, 4, 4, 4), 5),
        ],
    )
    def test_number_of_levels_correct(self, number_of_levels, channel_list, expected_levels):
        layer_instance = Decoder(
            number_of_levels=number_of_levels, channel_list=channel_list, filters=4, number_of_blocks=1
        )
        layer_instance.build(tf.TensorShape((1, 16, 16, 4)))
        assert len(layer_instance.decoderBlocks) == len(layer_instance.upSampleBlocks) == expected_levels

    @pytest.mark.parametrize("use_self_attention", [(True, False)])
    def test_self_attention_is_not_none(self, use_self_attention):
        layer_instance = Decoder(
            use_self_attention=use_self_attention, number_of_levels=1, filters=4, number_of_blocks=1
        )
        layer_instance.build(tf.TensorShape((1, 16, 16, 4)))
        if use_self_attention:
            assert layer_instance.SA is not None
        else:
            assert layer_instance.SA is None

    @pytest.mark.parametrize(
        ("number_of_levels", "channel_list", "filters", "limit_filters", "expected_channel_list"),
        [
            (1, None, 3, 128, [3]),
            (3, None, 8, 128, [32, 16, 8]),
            (6, None, 8, 128, [128, 128, 64, 32, 16, 8]),
            (6, [256, 32, 16, 8], 8, 128, [256, 32, 16, 8]),
        ],
    )
    def test_channel_list_as_expected(
        self, number_of_levels, channel_list, filters, limit_filters, expected_channel_list
    ):
        layer_instance = Decoder(
            number_of_levels=number_of_levels,
            limit_filters=limit_filters,
            channel_list=channel_list,
            filters=filters,
            number_of_blocks=1,
        )
        layer_instance.build(tf.TensorShape((1, 16, 16, 4)))
        assert layer_instance.channel_list == expected_channel_list

    @pytest.mark.xfail(reason="Test not implemented yet")
    @pytest.mark.parametrize(
        ("input_shape", "expected_shape"),
        [],
    )
    def test_skips_input_expected_shape(self, input_shape, expected_shape):
        ...
