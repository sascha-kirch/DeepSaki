import os

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors
import tensorflow as tf

from DeepSaki.losses.diffusion_model_losses import DiffusionLoss

#TODO: mock GaussianDiffusionProcess

#@pytest.mark.parametrize("batch_size",[1,8])
class TestDiffusionLoss:

    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_global_batch_size_as_expected(self):
        ...

    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_get_loss_weighting_returns_correct_type(self):
        ...
        # TODO: mock self.loss_weighting_type

    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_get_loss_weighting_returns_correct_shape(self):
        ...
        # TODO: mock self.loss_weighting_type

    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_get_loss_weighting_returns_raises_error(self):
        ...
        # TODO: mock self.loss_weighting_type

    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_simple_loss_expected_result(self):
        ...
        # TODO: mock _get_loss_weighting

    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_simple_loss_expected_shape(self):
        ...
        # TODO: mock _get_loss_weighting

    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_vlb_loss_expected_result(self):
        ...
        # TODO: mock get_vlb_loss_term

    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_vlb_loss_expected_shape(self):
        ...
        # TODO: mock get_vlb_loss_term

    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_hybrid_loss_expected_result(self):
        ...
        # TODO: mock simple_loss
        # TODO: mock vlb_loss
        # TODO: mock self.lambda_vlb

    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_hybrid_loss_expected_shape(self):
        ...
        # TODO: mock simple_loss
        # TODO: mock vlb_loss
        # TODO: mock self.lambda_vlb

    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_get_vlb_loss_term_returns_correct_value(self):
        ...
        # TODO: mock self.diffusion_process.q_xtm1_given_x0_xt_mean_var
        # TODO: mock self.diffusion_process.p_xtm1_given_xt_mean_var
        # TODO: mock calc_log_likelihood_of_discretized_gaussian
        # TODO: mock calc_kl_divergence_of_univariate_normal_distribution

    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_get_vlb_loss_term_returns_correct_shape(self):
        ...
        # TODO: mock self.diffusion_process.q_xtm1_given_x0_xt_mean_var
        # TODO: mock self.diffusion_process.p_xtm1_given_xt_mean_var
        # TODO: mock calc_log_likelihood_of_discretized_gaussian
        # TODO: mock calc_kl_divergence_of_univariate_normal_distribution

    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_get_vlb_prior_correct_shape(self):
        ...
        # TODO: mock self.diffusion_process.q_xt_given_x0_mean_var
        # TODO: mock calc_kl_divergence_of_univariate_normal_distribution

    @pytest.mark.xfail(reason="Test not implemented yet.")
    def test_get_vlb_prior_correct_shape(self):
        ...
        # TODO: mock self.diffusion_process.q_xt_given_x0_mean_var
        # TODO: mock calc_kl_divergence_of_univariate_normal_distribution
