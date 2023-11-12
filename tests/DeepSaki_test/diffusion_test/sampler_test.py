import os

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf

from DeepSaki.diffusion.sampler import SamplerResult, Sampler, DDPMSampler, DDIMSampler

class TestSamplerResult:
    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_shape_as_expected(self):
        ...

    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_type_as_expected(self):
        ...

class TestSampler:
    ...

class TestDDPMSampler:
    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_sampling_result_correct_shape(self):
        ...

class TestDDIMSampler:
    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_sample_method(self):
        ...

    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_sampling_steps_valid(self):
        ...

    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_sampling_result_correct_shape(self):
        ...

    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_ddim_sample(self):
        ...
