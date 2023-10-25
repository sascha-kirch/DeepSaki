import pytest
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf

from DeepSaki.constraints.constraints import NonNegative


@pytest.fixture(scope="class")
def non_negative_constraint():
    return NonNegative()


class TestNonNegativeConstraint:
    @pytest.mark.parametrize(
        "input_dtype",
        [
            tf.float16,
            tf.bfloat16,
            tf.float32,
            tf.float64,
        ],
    )
    def test_dtype_output_same_as_input(self, non_negative_constraint, input_dtype):
        output = non_negative_constraint(tf.constant([1, 2, 3], dtype=input_dtype))
        assert output.dtype == input_dtype

    @pytest.mark.parametrize(
        "input_tensor",
        [
            tf.constant([1.0]),
            tf.constant([1.0, 1.0]),
            tf.constant([[1.0, 1.0], [1.0, 1.0]]),
            tf.constant([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]),
        ],
    )
    def test_shape_output_same_as_input(self, non_negative_constraint, input_tensor):
        output = non_negative_constraint(input_tensor)
        assert output.shape == input_tensor.shape

    @pytest.mark.parametrize(
        ("input", "expected"),
        [
            (tf.constant([1.0]), tf.constant([1.0])),
            (tf.constant([-1.0]), tf.constant([0.0])),
            (tf.constant([1.0, 1.0]), tf.constant([1.0, 1.0])),
            (tf.constant([1.0, -1.0]), tf.constant([1.0, 0.0])),
            (tf.constant([-1.0, -1.0]), tf.constant([0.0, 0.0])),
            (tf.constant([[1.0, 1.0], [1.0, 1.0]]), tf.constant([[1.0, 1.0], [1.0, 1.0]])),
            (tf.constant([[-1.0, -1.0], [1.0, 1.0]]), tf.constant([[0.0, 0.0], [1.0, 1.0]])),
            (tf.constant([[1.0, -1.0], [-1.0, 1.0]]), tf.constant([[1.0, 0.0], [0.0, 1.0]])),
        ],
    )
    def test_output_is_correct(self, non_negative_constraint, input, expected):
        output = non_negative_constraint(input)
        assert tf.math.reduce_all(output == expected)
