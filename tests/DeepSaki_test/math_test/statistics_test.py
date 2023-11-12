import os

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors


class TestApproximatedStandardNormalCdf:
    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_approximation_is_good(self):
        ...


class TestLogLikelihoodOfDiscretizedGaussian:
    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_result_as_expected(self):
        ...


class TestKlDivergenceOfUnivariateNormalDistribution:
    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_result_as_expected(self):
        ...
