import os

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf

from DeepSaki.initializers.initializer_helper import make_initializer_complex

@pytest.mark.parametrize(
    "initializer",
    [
        tf.initializers.HeNormal(seed=123),
        tf.keras.initializers.RandomNormal(seed=123),
        tf.initializers.GlorotUniform(seed=123),
        tf.initializers.Zeros(),
    ],
)
@pytest.mark.parametrize("dtype", [tf.complex64, tf.complex128])
@pytest.mark.parametrize("shape", [(2, 2), (3, 2, 1), (1, 2, 3, 4)])
class TestMakeInitializerComplex:
    def _get_initializer_values(self, initializer, dtype, shape):
        complex_initializer = make_initializer_complex(initializer)
        return complex_initializer(shape, dtype)

    def test_initializer_values_complex(self, initializer, dtype, shape):
        values = self._get_initializer_values(initializer, dtype, shape)
        assert values.dtype == dtype

    def test_initializer_values_correct_shape(self, initializer, dtype, shape):
        values = self._get_initializer_values(initializer, dtype, shape)
        assert values.shape == shape
