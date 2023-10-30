import os
from typing import Callable

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors

import tensorflow as tf

def _mock_downsample_block(mocker, calling_module) -> None:
    """Mocks the DeepSaki.layers.DownSampleBlock() layer."""

    def _mocked_class(**kwargs) -> Callable[[tf.Tensor], tf.Tensor]:
        def _mocked_call(input_tensor: tf.Tensor) -> tf.Tensor:
            input_shape = input_tensor.shape

            expected_shape = (
                input_shape[0],
                input_shape[1] // 2,
                input_shape[2] // 2,
                input_shape[3],
            )
            return tf.ones(shape=expected_shape)

        return _mocked_call

    mock_encoder = mocker.patch(f"{calling_module}.DownSampleBlock")
    mock_encoder.side_effect = _mocked_class


def _mock_upsample_block(mocker, calling_module) -> None:
    """Mocks the DeepSaki.layers.UpSampleBlock() layer."""

    def _mocked_class(**kwargs) -> Callable[[tf.Tensor], tf.Tensor]:
        def _mocked_call(input_tensor: tf.Tensor) -> tf.Tensor:
            input_shape = input_tensor.shape

            expected_shape = (
                input_shape[0],
                input_shape[1] * 2,
                input_shape[2] * 2,
                input_shape[3],
            )
            return tf.ones(shape=expected_shape)

        return _mocked_call

    mock_encoder = mocker.patch(f"{calling_module}.UpSampleBlock")
    mock_encoder.side_effect = _mocked_class


def _mock_residualblock(mocker, calling_module) -> None:
    """Mocks the DeepSaki.layers.ResidualBlock() layer."""

    def _mocked_class(**kwargs) -> Callable[[tf.Tensor], tf.Tensor]:
        def _mocked_call(input_tensor: tf.Tensor) -> tf.Tensor:
            input_shape = input_tensor.shape
            expected_shape = [*input_shape[0:-1], kwargs["filters"]]
            return tf.ones(shape=expected_shape)

        return _mocked_call

    mock_encoder = mocker.patch(f"{calling_module}.ResidualBlock")
    mock_encoder.side_effect = _mocked_class


def _mock_resblock_down(mocker, calling_module) -> None:
    """Mocks the DeepSaki.layers.ResBlockDown() layer."""

    def _mocked_class(**kwargs) -> Callable[[tf.Tensor], tf.Tensor]:
        def _mocked_call(input_tensor: tf.Tensor) -> tf.Tensor:
            input_shape = input_tensor.shape

            expected_shape = (
                input_shape[0],
                input_shape[1] // 2,
                input_shape[2] // 2,
                input_shape[3],
            )
            return tf.ones(shape=expected_shape)

        return _mocked_call

    mock_encoder = mocker.patch(f"{calling_module}.ResBlockDown")
    mock_encoder.side_effect = _mocked_class


def _mock_resblock_up(mocker, calling_module) -> None:
    """Mocks the DeepSaki.layers.ResBlockUp() layer."""

    def _mocked_class(**kwargs) -> Callable[[tf.Tensor], tf.Tensor]:
        def _mocked_call(input_tensor: tf.Tensor) -> tf.Tensor:
            input_shape = input_tensor.shape

            expected_shape = (
                input_shape[0],
                input_shape[1] * 2,
                input_shape[2] * 2,
                input_shape[3],
            )
            return tf.ones(shape=expected_shape)

        return _mocked_call

    mock_encoder = mocker.patch(f"{calling_module}.ResBlockUp")
    mock_encoder.side_effect = _mocked_class


def _mock_scale_layer(mocker, calling_module) -> None:
    """Mocks the DeepSaki.layers.ScaleLayer() layer."""

    def _mocked_class(**kwargs) -> Callable[[tf.Tensor], tf.Tensor]:
        def _mocked_call(input_tensor: tf.Tensor) -> tf.Tensor:
            input_shape = input_tensor.shape
            expected_shape = input_shape
            return tf.ones(shape=expected_shape)

        return _mocked_call

    mock_encoder = mocker.patch(f"{calling_module}.ScaleLayer")
    mock_encoder.side_effect = _mocked_class


def _mock_scalar_gated_self_attention(mocker, calling_module) -> None:
    """Mocks the DeepSaki.layers.ScalarGatedSelfAttention() layer."""

    def _mocked_class(**kwargs) -> Callable[[tf.Tensor], tf.Tensor]:
        def _mocked_call(input_tensor: tf.Tensor) -> tf.Tensor:
            input_shape = input_tensor.shape
            expected_shape = input_shape
            return tf.ones(shape=expected_shape)

        return _mocked_call

    mock_encoder = mocker.patch(f"{calling_module}.ScalarGatedSelfAttention")
    mock_encoder.side_effect = _mocked_class


def _mock_global_sum_pooling_2d(mocker, calling_module) -> None:
    """Mocks the DeepSaki.layers.GlobalSumPooling2D() layer."""

    def _mocked_class(**kwargs) -> Callable[[tf.Tensor], tf.Tensor]:
        def _mocked_call(input_tensor: tf.Tensor) -> tf.Tensor:
            input_shape = input_tensor.shape
            expected_shape = [input_shape[0], input_shape[-1]]
            return tf.ones(shape=expected_shape)

        return _mocked_call

    mock_encoder = mocker.patch(f"{calling_module}.GlobalSumPooling2D")
    mock_encoder.side_effect = _mocked_class


def _mock_encoder(mocker, calling_module) -> None:
    """Mocks the DeepSaki.layers.Encoder() layer."""

    def _mocked_class(**kwargs) -> Callable[[tf.Tensor], tf.Tensor]:
        """The mocked Encoder() class.

        Args:
            kwargs: Keyword args passed to the initializer object of the Encoder(). Contains all arguments passed when
                instanciating the encoder

        Returns:
            function that mocks the calling the object. Returns a tensor of ones of the expected shape.
        """

        def _mocked_call(input_tensor: tf.Tensor) -> tf.Tensor:
            input_shape = input_tensor.shape
            if "channel_list" in kwargs and kwargs["channel_list"] is not None:
                expected_output_filters = kwargs["channel_list"][-1]
            else:
                expected_output_filters = min(
                    kwargs["filters"] * 2 ** (kwargs["number_of_levels"] - 1), kwargs["limit_filters"]
                )

            expected_shape = (
                input_shape[0],
                input_shape[1] // 2 ** kwargs["number_of_levels"],
                input_shape[2] // 2 ** kwargs["number_of_levels"],
                expected_output_filters,
            )
            output = tf.ones(shape=expected_shape)
            if "output_skips" in kwargs and kwargs["output_skips"]:
                output = (output, None)  # We don't care about the skip connections during autoencoder test.
            return output

        return _mocked_call

    mock_encoder = mocker.patch(f"{calling_module}.Encoder")
    mock_encoder.side_effect = _mocked_class


def _mock_bottleneck(mocker, calling_module) -> None:
    """Mocks the DeepSaki.layers.Bottleneck() layer."""

    def _mocked_class(**kwargs) -> Callable[[tf.Tensor], tf.Tensor]:
        def _mocked_call(input_tensor: tf.Tensor) -> tf.Tensor:
            input_shape = input_tensor.shape
            expected_shape = input_shape
            if "channel_list" in kwargs and kwargs["channel_list"] is not None:
                expected_shape = (
                    input_shape[0],
                    input_shape[1],
                    input_shape[2],
                    kwargs["channel_list"][-1],
                )
            else:
                expected_shape = input_shape

            return tf.ones(shape=expected_shape)

        return _mocked_call

    mock_bottleneck = mocker.patch(f"{calling_module}.Bottleneck")
    mock_bottleneck.side_effect = _mocked_class


def _mock_decoder(mocker, calling_module) -> None:
    """Mocks the DeepSaki.layers.Decoder() layer."""

    def _mocked_class(**kwargs) -> Callable[[tf.Tensor], tf.Tensor]:
        def _mocked_call(input_tensor: tf.Tensor) -> tf.Tensor:
            if "enable_skip_connections_input" in kwargs and kwargs["enable_skip_connections_input"]:
                input_tensor = input_tensor[0]  # We don't care about the skip connections during autoencoder test.

            if "channel_list" in kwargs and kwargs["channel_list"] is not None:
                expected_output_filters = kwargs["channel_list"][-1]
            else:
                expected_output_filters = kwargs["filters"]
            input_shape = input_tensor.shape
            expected_shape = (
                input_shape[0],
                input_shape[1] * 2 ** kwargs["number_of_levels"],
                input_shape[2] * 2 ** kwargs["number_of_levels"],
                expected_output_filters,
            )
            return tf.ones(shape=expected_shape)

        return _mocked_call

    mock_decoder = mocker.patch(f"{calling_module}.Decoder")
    mock_decoder.side_effect = _mocked_class
