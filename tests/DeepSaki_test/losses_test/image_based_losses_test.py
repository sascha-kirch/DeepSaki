import pytest
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf
import numpy as np

from DeepSaki.losses.image_based_losses import PixelDistanceLoss, StructuralSimilarityLoss

class TestPixelDistanceLoss:

    @pytest.mark.parametrize(
    "calculation_type, loss_type",
    [
        ("Wrong String", "mae"),
        ("per_image", "Wrong String"),
        ("Wrong String", "Wrong String"),
        (3, "mae"),
        (5.6, "mae"),
        (None, "mae"),
        ("per_image", 2),
        ("per_image", 2.0),
        ("per_image", None),
    ],
)
    def test_init_raises_error(self,calculation_type, loss_type):
        with pytest.raises(ValueError):
            _ = PixelDistanceLoss(global_batch_size=64, calculation_type=calculation_type, loss_type=loss_type)

    @pytest.mark.parametrize(
        "loss_type, expected",[
            ("mae", tf.abs),
            ("mse", tf.square),
        ]
    )
    def test_init_correct_error_func_selected(self, loss_type, expected):
        loss = PixelDistanceLoss(global_batch_size=64, loss_type=loss_type)
        assert loss.error_func_type == expected

    @pytest.mark.skip(reason="Not implemented yet.")
    def test_init_correct_loss_calc_func_selected(self):
        ...

    @pytest.mark.skip(reason="Not implemented yet.")
    def test_calc_loss_per_image(self):
        ...

    @pytest.mark.skip(reason="Not implemented yet.")
    def test_calc_loss_per_channel(self):
        ...

    def test_isinstance_tensorflow_loss(self):
        loss = PixelDistanceLoss(global_batch_size=64)
        assert isinstance(loss, tf.keras.losses.Loss)

class TestSsimLoss:

    @pytest.mark.parametrize(
    "calculation_type",
    [
        "Wrong String",
        3,
        5.6,
        None,
    ],
)
    def test_init_raises_error(self,calculation_type):
        with pytest.raises(ValueError):
            _ = StructuralSimilarityLoss(global_batch_size=64, calculation_type=calculation_type)

    @pytest.mark.skip(reason="Not implemented yet.")
    def test_init_correct_error_func_selected(self):
        ...

    @pytest.mark.skip(reason="Not implemented yet.")
    def test_init_correct_loss_calc_func_selected(self):
        ...

    @pytest.mark.skip(reason="Not implemented yet.")
    def test_calc_loss_per_image(self):
        ...

    @pytest.mark.skip(reason="Not implemented yet.")
    def test_calc_loss_per_channel(self):
        ...

    def test_isinstance_tensorflow_loss(self):
        loss = StructuralSimilarityLoss(global_batch_size=64)
        assert isinstance(loss, tf.keras.losses.Loss)
