import os

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf

from DeepSaki.augmentations.grid_cutting import _get_mask
from DeepSaki.augmentations.grid_cutting import _invert_mask
from DeepSaki.augmentations.grid_cutting import _random_boundingbox


class TestRandomBoundingBox:
    @pytest.mark.parametrize("height", [10, 50, 100])
    @pytest.mark.parametrize("width", [10, 50, 100])
    def test_random_bounding_box_smaller_than_width_height(self, height, width):
        # I loop because I hope to cover the randomness of the function...
        for _ in range(1000):
            x1, y1, x2, y2 = _random_boundingbox(height, width)
            assert (x2 - x1) <= width
            assert (y2 - y1) <= height

    @pytest.mark.parametrize("height", [10, 50, 100])
    @pytest.mark.parametrize("width", [10, 50, 100])
    def test_random_bounding_box_dtype_int(self, height, width):
        # I loop because I hope to cover the randomness of the function...
        for _ in range(1000):
            x1, x2, y1, y2 = _random_boundingbox(height, width)
            assert all(
                [isinstance(point, int) for point in [x1, x2, y1, y2]]
            ), f"Dtypes are:{[type(point) for point in [x1, x2, y1, y2]]}"


class TestGetMask:
    @pytest.mark.parametrize("batch", [1, 8, 16])
    @pytest.mark.parametrize("height", [32, 50, 128])
    @pytest.mark.parametrize("width", [32, 50, 128])
    @pytest.mark.parametrize("channel", [32, 50, 128])
    def test_get_mask_shape_equal_input_shape(self, batch, height, width, channel):
        result = _get_mask([batch, height, width, channel])
        assert result.shape == [batch, height, width, channel]

    @pytest.mark.parametrize(
        ("input_shape", "expected_dtype"),
        [
            ([8, 512, 512, 3], tf.float32),
            ([16, 256, 256, 64], tf.float32),
            ([1, 150, 150, 30], tf.float32),
        ],
    )
    def test_get_mask_dtype(self, input_shape, expected_dtype):
        result = _get_mask(input_shape)
        assert result.dtype == expected_dtype


class TestInvertMask:
    @pytest.mark.parametrize(
        ("input", "expected"),
        [
            (tf.constant([1, 0]), tf.constant([0, 1])),
            (tf.constant([[1, 0], [1, 0]]), tf.constant([[0, 1], [0, 1]])),
        ],
    )
    def test_invert_mask(self, input, expected):
        result = _invert_mask(input)
        assert tf.math.reduce_all(result.numpy() == pytest.approx(expected.numpy(), 0.01))


@pytest.mark.skip(reason="Not implemented yet.")
class TestCutMix:
    pass


@pytest.mark.skip(reason="Not implemented yet.")
class TestCutOut:
    pass
