import os

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf

from DeepSaki.utils.environment import detect_accelerator
from DeepSaki.utils.environment import enable_mixed_precision
from DeepSaki.utils.environment import enable_xla_acceleration


class TestEnvironment:
    @pytest.fixture()
    def _tf_policy_restorer(self):
        policy = tf.keras.mixed_precision.global_policy()
        yield
        tf.keras.mixed_precision.set_global_policy(policy)

    def _mock_tpu(self, mocker, tpu_available):
        mock_tpu = mocker.patch("DeepSaki.utils.environment.tf.distribute.cluster_resolver.TPUClusterResolver")
        if tpu_available:
            mock_tpu.return_value = True
            mock_tpu = mocker.patch("DeepSaki.utils.environment.tf.distribute.cluster_resolver.TPUClusterResolver")
        else:
            mock_tpu.side_effect = ValueError

    @pytest.mark.xfail(reason="Not yet able to fully mock GPU and TPU")
    @pytest.mark.parametrize(
        ("tpu_available", "gpu_memory_groth", "gpus", "expected_strategy", "expected_runtimme_env"),
        [
            (True, False, 0, tf.distribute.TPUStrategy, "TPU"),
            (False, False, 1, tf.distribute.Strategy, "GPU"),
            (False, False, 2, tf.distribute.MirroredStrategy, "GPU"),
            (False, False, 0, tf.distribute.Strategy, "CPU"),
        ],
    )
    def test_detect_accelerator(
        self, mocker, tpu_available, gpu_memory_groth, gpus, expected_strategy, expected_runtimme_env
    ):
        self._mock_tpu(mocker, tpu_available)
        strategy, runtime_environment, _ = detect_accelerator(gpu_memory_groth)

        assert strategy == expected_strategy
        assert runtime_environment == expected_runtimme_env

    @pytest.mark.parametrize(
        ("enable", "expected"),
        [
            (False, ""),
            (True, "autoclustering"),
        ],
    )
    def test_enable_xla_acceleration(self, enable, expected):
        if enable:
            enable_xla_acceleration()
        assert tf.config.optimizer.get_jit() == expected

    @pytest.mark.parametrize(
        ("enable", "tpu_available", "expected"),
        [
            (False, False, "float32"),
            (True, False, "mixed_float16"),
            (False, True, "float32"),
            pytest.param(
                True, True, "mixed_bfloat16", marks=pytest.mark.xfail(reason="Havn't figured out how to mock TPU call.")
            ),
        ],
    )
    @pytest.mark.usefixtures("_tf_policy_restorer")
    def test_enable_mixed_precision(self, mocker, enable, tpu_available, expected):
        self._mock_tpu(mocker, tpu_available)

        if enable:
            enable_mixed_precision()
        policy = tf.keras.mixed_precision.global_policy()
        assert policy.name == expected
