import pytest
import os
import inspect
from contextlib import nullcontext as does_not_raise

from unittest import mock

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf
import numpy as np

from DeepSaki.utils.environment import detect_accelerator, enable_mixed_precision, enable_xla_acceleration

@pytest.fixture
def tf_policy_restorer():
    policy = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(policy)


@pytest.mark.parametrize("enable, expected",[
    (False, ""),
    (True, "autoclustering"),
])
def test_enable_xla_acceleration(enable, expected):
    if enable:
        enable_xla_acceleration()
    assert tf.config.optimizer.get_jit() == expected

@pytest.mark.parametrize("enable, tpu, expected",[
    (False, False, "float32"),
    (True, False, "mixed_float16"),
    (False, True, "float32"),
    pytest.param(True, True, "mixed_bfloat16", marks=pytest.mark.xfail(reason="Havn't figured out how to mock TPU call.")),
])
@pytest.mark.usefixtures("tf_policy_restorer")
def test_enable_mixed_precision(enable, tpu, expected):
    if tpu:
        mock.patch("DeepSaki.utils.tf.distribute.cluster_resolver.TPUClusterResolver()", True)
        #mocker.patch("tf.distribute.cluster_resolver.TPUClusterResolver()", True)
    if enable:
        enable_mixed_precision()
    policy = tf.keras.mixed_precision.global_policy()
    assert policy.name == expected
