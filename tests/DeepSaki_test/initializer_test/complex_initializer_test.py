import os

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf

from DeepSaki.initializers.complex_initializer import ComplexInitializer


@pytest.mark.parametrize(
    ("initializer_imag"),
    [
        None,
        tf.initializers.HeNormal(seed=123),
        tf.keras.initializers.RandomNormal(seed=123),
        tf.initializers.GlorotUniform(seed=123),
    ],
)
@pytest.mark.parametrize(
    ("initializer_real"),
    [
        (tf.initializers.HeNormal(seed=456)),
        (tf.keras.initializers.RandomNormal(seed=456)),
        (tf.initializers.GlorotUniform(seed=456)),
    ],
)
@pytest.mark.parametrize("dtype", [tf.complex64, tf.complex128])
@pytest.mark.parametrize("shape", [(2, 2), (3, 2, 1), (1, 2, 3, 4)])
class TestComplexInitializer:
    def _get_initializer_values(self, initializer_real, initializer_imag, dtype, shape):
        complex_initializer = ComplexInitializer(initializer_real, initializer_imag)
        return complex_initializer(shape, dtype)

    def test_initializer_values_complex(self, initializer_real, initializer_imag, dtype, shape):
        values = self._get_initializer_values(initializer_real, initializer_imag, dtype, shape)
        assert values.dtype == dtype

    def test_initializer_values_correct_shape(self, initializer_real, initializer_imag, dtype, shape):
        values = self._get_initializer_values(initializer_real, initializer_imag, dtype, shape)
        assert values.shape == shape

    def test_real_imag_same_if_single_initializer(self, initializer_real, initializer_imag, dtype, shape):
        values = self._get_initializer_values(initializer_real, initializer_imag, dtype, shape)

        assert tf.math.reduce_all(
            (tf.math.real(values).numpy() == pytest.approx(tf.math.imag(values).numpy(), 0.01))
            == (initializer_imag is None)
        )
