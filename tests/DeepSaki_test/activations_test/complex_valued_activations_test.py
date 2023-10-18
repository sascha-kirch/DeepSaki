import pytest
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf

from DeepSaki.activations.complex_valued_activations import ComplexActivation


class TestComplexActivation:
    @pytest.mark.parametrize(
        "activation, input, expected",
        [
            (tf.keras.layers.ReLU(), tf.constant([1.0, -1.0]), tf.complex(real=[1.0, 0.0], imag=[0.0, 0.0])),
            (tf.keras.layers.ReLU(), tf.constant([1.0, 1.0]), tf.complex(real=[1.0, 1.0], imag=[0.0, 0.0])),
            (tf.keras.layers.ReLU(), tf.constant([-1.0, -1.0]), tf.complex(real=[0.0, 0.0], imag=[0.0, 0.0])),
            (
                tf.keras.layers.ReLU(),
                tf.complex(real=[1.0, 0.0], imag=[1.0, 0.0]),
                tf.complex(real=[1.0, 0.0], imag=[1.0, 0.0]),
            ),
            (
                tf.keras.layers.ReLU(),
                tf.complex(real=[1.0, -1.0], imag=[1.0, -1.0]),
                tf.complex(real=[1.0, 0.0], imag=[1.0, 0.0]),
            ),
            (
                tf.keras.layers.ReLU(),
                tf.complex(real=[-1.0, 1.0], imag=[-1.0, 1.0]),
                tf.complex(real=[0.0, 1.0], imag=[0.0, 1.0]),
            ),
        ],
    )
    def test_call_has_expected_output(self, activation, input, expected):
        activation = ComplexActivation(activation=activation)
        output = activation(input)
        assert tf.math.reduce_all(output == expected)

    @pytest.mark.parametrize(
        "input",
        [
            (tf.constant([1.0, 1.0], dtype=tf.float16)),
            (tf.constant([1.0, 1.0], dtype=tf.float32)),
            (tf.constant([1.0, 1.0], dtype=tf.float64)),
            (tf.constant([1.0, 1.0], dtype=tf.bfloat16)),
            (
                tf.complex(
                    real=tf.constant([1.0, 1.0], dtype=tf.float32), imag=tf.constant([1.0, 1.0], dtype=tf.float32)
                )
            ),
            (
                tf.complex(
                    real=tf.constant([1.0, 1.0], dtype=tf.float64), imag=tf.constant([1.0, 1.0], dtype=tf.float64)
                )
            ),
        ],
    )
    def test_dtype(self, input):
        activation = ComplexActivation()
        output = activation(input)
        assert output.dtype in (tf.dtypes.complex64, tf.dtypes.complex128)

    @pytest.mark.parametrize(
        "input",
        [
            (tf.constant([1.0])),
            (tf.constant([1.0, 1.0])),
            (tf.constant([[1.0, 1.0], [1.0, 1.0]])),
            (tf.constant([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])),
            (tf.complex(real=[1.0, 1.0], imag=[1.0, 1.0])),
            (tf.complex(real=[[1.0, 1.0], [1.0, 1.0]], imag=[[1.0, 1.0], [1.0, 1.0]])),
            (
                tf.complex(
                    real=[[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                    imag=[[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
                )
            ),
        ],
    )
    def test_input_shape_equals_output_shape(self, input):
        activation = ComplexActivation()
        output = activation(input)
        assert output.shape == input.shape
