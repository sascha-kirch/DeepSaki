import os

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf

from DeepSaki.losses.image_based_losses import ImageBasedLoss
from DeepSaki.losses.image_based_losses import PixelDistanceLoss
from DeepSaki.losses.image_based_losses import StructuralSimilarityLoss
from DeepSaki.losses.image_based_losses import LossCalcType
from DeepSaki.losses.image_based_losses import LossType


class TestImageBasedLossAbstractBase:
    @pytest.mark.parametrize(
        "calculation_type",
        [
            "Wrong String",
            3,
            5.6,
            None,
        ],
    )
    def test_init_raises_value_error(self, mocker, calculation_type):
        mocker.patch.multiple(ImageBasedLoss, __abstractmethods__=set())
        with pytest.raises(ValueError):
            _ = ImageBasedLoss(global_batch_size=64, calculation_type=calculation_type)

    @pytest.mark.parametrize(
        ("calculation_type", "expected"),
        [
            (LossCalcType.PER_CHANNEL, "_calc_loss_per_channel"),
            (LossCalcType.PER_IMAGE, "_calc_loss_per_image"),
        ],
    )
    def test_correct_loss_calc_func_selected(self, mocker, calculation_type, expected):
        # arange - Mock abstract class
        mocker.patch.multiple(ImageBasedLoss, __abstractmethods__=set())
        instance = ImageBasedLoss(global_batch_size=1, calculation_type=calculation_type)

        # hacky way of checking. reason: due to the mocking it is not possible to compare the functions
        assert instance.loss_calc_func.__name__ == expected

    @pytest.mark.parametrize(
        ("img1", "img2", "global_batch_size"),
        [
            (tf.ones(shape=(1, 1, 1, 4)), tf.ones(shape=(1, 1, 1, 4)), 1),
            (tf.ones(shape=(1, 1, 1, 3)), tf.ones(shape=(1, 1, 1, 3)), 64),
            (tf.ones(shape=(1, 64, 64, 4)), tf.ones(shape=(1, 64, 64, 4)), 64),
            (tf.ones(shape=(1, 10, 10, 4)), tf.ones(shape=(1, 10, 10, 4)), 50),
            (tf.ones(shape=(1, 99, 99, 3)), tf.ones(shape=(1, 99, 99, 3)), 87),
        ],
    )
    def test_calc_loss_per_image(self, mocker, img1, img2, global_batch_size):
        # arange - Mock abstract class and define error_func
        mocker.patch.multiple(ImageBasedLoss, __abstractmethods__=set())
        instance = ImageBasedLoss(global_batch_size=global_batch_size)
        error_func = lambda x, y: x * y * global_batch_size

        # act
        result = instance._calc_loss_per_image(img1, img2, error_func)
        expected = tf.ones_like(img1)

        # assert
        assert tf.math.reduce_all(result.numpy() == pytest.approx(expected.numpy(), 0.01))

    @pytest.mark.parametrize(
        ("img1", "img2", "global_batch_size", "channel_weights"),
        [
            (tf.ones(shape=(1, 1, 1, 4)), tf.ones(shape=(1, 1, 1, 4)), 1, [1, 1, 1, 3]),
            (tf.ones(shape=(1, 1, 1, 3)), tf.ones(shape=(1, 1, 1, 3)), 64, [1, 1, 2]),
            (tf.ones(shape=(1, 64, 64, 4)), tf.ones(shape=(1, 64, 64, 4)), 64, [1, 1, 1, 1]),
            (tf.ones(shape=(1, 10, 10, 4)), tf.ones(shape=(1, 10, 10, 4)), 50, [1, 1, 1, 3]),
            (tf.ones(shape=(1, 99, 99, 3)), tf.ones(shape=(1, 99, 99, 3)), 87, [1, 1, 2]),
        ],
    )
    def test_calc_loss_per_channel(self, mocker, img1, img2, global_batch_size, channel_weights):
        # arange - Mock abstract class and _get_channel_weights method and define error_func
        mocker.patch.multiple(ImageBasedLoss, __abstractmethods__=set())
        instance = ImageBasedLoss(global_batch_size=global_batch_size)
        mocker.patch(
            "DeepSaki.losses.image_based_losses.ImageBasedLoss._get_channel_weights", return_value=channel_weights
        )
        error_func = lambda x, y: x * y * global_batch_size

        # act
        result = instance._calc_loss_per_channel(img1, img2, error_func)
        expected = tf.reduce_sum(tf.ones_like(img1), axis=-1) + (channel_weights[-1] - 1)

        # assert
        assert tf.math.reduce_all(result.numpy() == pytest.approx(expected.numpy(), 0.01))

    @pytest.mark.parametrize(
        ("img_shape", "normalize_last_channel", "expected"),
        [
            (tf.TensorShape([1, 64, 64, 3]), False, [1, 1, 1]),
            (tf.TensorShape([1, 64, 64, 3]), True, [1, 1, 2]),
            (tf.TensorShape([1, 64, 64, 4]), False, [1, 1, 1, 1]),
            (tf.TensorShape([1, 64, 64, 4]), True, [1, 1, 1, 3]),
        ],
    )
    def test_get_channel_weights(self, mocker, img_shape, normalize_last_channel, expected):
        # arange - Mock abstract class
        mocker.patch.multiple(ImageBasedLoss, __abstractmethods__=set())
        instance = ImageBasedLoss(global_batch_size=1)
        # Act
        result = instance._get_channel_weights(img_shape, normalize_last_channel)
        # assert
        assert tf.math.reduce_all(result == pytest.approx(expected, 0.01))

    @pytest.mark.skip(reason="Not implemented yet.")
    def test_call(self):
        ...


class TestPixelDistanceLoss:
    @pytest.mark.parametrize(
        "loss_type",
        [
            "Wrong String",
            2,
            2.0,
            None,
        ],
    )
    def test_init_raises_error(self, loss_type):
        with pytest.raises(ValueError):
            _ = PixelDistanceLoss(global_batch_size=64, loss_type=loss_type)

    @pytest.mark.parametrize(
        ("loss_type", "expected"),
        [
            (LossType.MAE, tf.abs),
            (LossType.MSE, tf.square),
        ],
    )
    def test_init_correct_error_func_selected(self, loss_type, expected):
        loss = PixelDistanceLoss(global_batch_size=64, loss_type=loss_type)
        assert loss.error_func_type == expected

    @pytest.mark.parametrize(
        ("tensor1", "tensor2", "loss_type", "expected"),
        [
            (3 * tf.ones(shape=[1, 1, 1, 1]), tf.ones(shape=[1, 1, 1, 1]), LossType.MAE, tf.constant(2.0)),
            (3 * tf.ones(shape=[8, 64, 64, 4]), tf.ones(shape=[8, 64, 64, 4]), LossType.MAE, tf.constant(2.0)),
            (5 * tf.ones(shape=[1, 1, 1, 1]), tf.ones(shape=[1, 1, 1, 1]), LossType.MAE, tf.constant(4.0)),
            (5 * tf.ones(shape=[8, 64, 64, 4]), tf.ones(shape=[8, 64, 64, 4]), LossType.MAE, tf.constant(4.0)),
            (3 * tf.ones(shape=[1, 1, 1, 1]), tf.ones(shape=[1, 1, 1, 1]), LossType.MSE, tf.constant(4.0)),
            (3 * tf.ones(shape=[8, 64, 64, 4]), tf.ones(shape=[8, 64, 64, 4]), LossType.MSE, tf.constant(4.0)),
            (5 * tf.ones(shape=[1, 1, 1, 1]), tf.ones(shape=[1, 1, 1, 1]), LossType.MSE, tf.constant(16.0)),
            (5 * tf.ones(shape=[8, 64, 64, 4]), tf.ones(shape=[8, 64, 64, 4]), LossType.MSE, tf.constant(16.0)),
        ],
    )
    def test_error_func(self, tensor1, tensor2, loss_type, expected):
        loss = PixelDistanceLoss(global_batch_size=64, loss_type=loss_type)
        result = loss._error_func(tensor1, tensor2)
        assert tf.math.reduce_all(result.numpy() == pytest.approx(expected.numpy(), 0.01))

    def test_isinstance_tensorflow_loss(self):
        loss = PixelDistanceLoss(global_batch_size=64)
        assert isinstance(loss, tf.keras.losses.Loss)


class TestSsimLoss:
    @pytest.mark.parametrize(
        ("tensor1", "tensor2", "expected"),
        [
            (tf.ones(shape=[1, 1, 1, 1]), tf.ones(shape=[1, 1, 1, 1]), tf.constant(0.0)),
            (tf.ones(shape=[8, 64, 64, 4]), tf.ones(shape=[8, 64, 64, 4]), tf.constant(0.0)),
            (tf.ones(shape=[1, 1, 1, 1]), tf.zeros(shape=[1, 1, 1, 1]), tf.constant(1.0)),
            (tf.ones(shape=[8, 64, 64, 4]), tf.zeros(shape=[8, 64, 64, 4]), tf.constant(1.0)),
        ],
    )
    def test_error_func(self, tensor1, tensor2, expected):
        loss = StructuralSimilarityLoss(global_batch_size=64)
        result = loss._error_func(tensor1, tensor2)
        assert tf.math.reduce_all(result.numpy() == pytest.approx(expected.numpy(), 0.01))

    def test_isinstance_tensorflow_loss(self):
        loss = StructuralSimilarityLoss(global_batch_size=64)
        assert isinstance(loss, tf.keras.losses.Loss)
