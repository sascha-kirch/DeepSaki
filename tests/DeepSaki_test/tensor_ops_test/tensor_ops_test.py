import os

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf

from DeepSaki.tensor_ops.tensor_ops import sample_array_to_tensor


@pytest.mark.parametrize(
    "array",
    [
        tf.constant([0, 1, 2, 3, 4, 5, 6]),
    ],
)
class TestSampleArrayToTensor:
    @pytest.mark.parametrize(
        ("shape", "index", "expected_shape"),
        [
            ((1, 1, 1, 1), 2, (1, 1, 1, 1)),
            ((-1, 1, 1), [3, 4, 5], (3, 1, 1)),
            ((1, 2), [1, 3], (1, 2)),
        ],
    )
    def test_output_shape_correct(self, array, index, shape, expected_shape):
        result = sample_array_to_tensor(array, index, shape)
        print(result)
        assert result.shape == expected_shape

    @pytest.mark.parametrize(
        "dtype",
        [
            tf.float32,
            tf.float64,
            tf.int32,
        ],
    )
    @pytest.mark.parametrize(
        "index",
        [
            1,
            [1, 3, 5],
        ],
    )
    def test_dtype_correct(self, array, index, dtype):
        result = sample_array_to_tensor(array, index, dtype=dtype)
        assert result.dtype == dtype
