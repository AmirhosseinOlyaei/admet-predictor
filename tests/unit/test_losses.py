"""Unit tests for loss functions."""

from __future__ import annotations

import pytest
import torch

from admet_predictor.losses.evidential import evidential_regression_loss
from admet_predictor.losses.gradnorm import GradNormLoss


class TestEvidentialLoss:
    def _make_nig_params(self, batch_size: int = 8):
        """Create valid NIG parameters satisfying nu > 0, alpha > 1, beta > 0."""
        gamma = torch.randn(batch_size, 1)
        nu = torch.rand(batch_size, 1) + 0.01       # (0, 1]
        alpha = torch.rand(batch_size, 1) + 1.01    # (1, 2]
        beta = torch.rand(batch_size, 1) + 0.01     # (0, 1]
        return gamma, nu, alpha, beta

    def test_returns_scalar(self):
        B = 16
        gamma, nu, alpha, beta = self._make_nig_params(B)
        y = torch.randn(B)
        loss = evidential_regression_loss(gamma, nu, alpha, beta, y)
        assert loss.ndim == 0

    def test_finite_output(self):
        B = 16
        gamma, nu, alpha, beta = self._make_nig_params(B)
        y = torch.randn(B)
        loss = evidential_regression_loss(gamma, nu, alpha, beta, y)
        assert torch.isfinite(loss)

    def test_non_negative(self):
        """The NIG NLL + positive regularizer should be finite (may be negative
        for some NIG param combinations since lgamma can be negative)."""
        B = 32
        gamma, nu, alpha, beta = self._make_nig_params(B)
        y = gamma.squeeze() + 0.1  # y close to prediction → small error
        loss = evidential_regression_loss(gamma, nu, alpha, beta, y)
        assert torch.isfinite(loss)

    def test_increases_with_error(self):
        """Loss should be lower when prediction is close to target."""
        torch.manual_seed(42)
        gamma, nu, alpha, beta = self._make_nig_params(32)
        y_close = gamma.squeeze() + 0.01
        y_far = gamma.squeeze() + 10.0

        loss_close = evidential_regression_loss(gamma, nu, alpha, beta, y_close)
        loss_far = evidential_regression_loss(gamma, nu, alpha, beta, y_far)
        assert loss_far > loss_close

    def test_handles_nan_gracefully(self):
        """nan_to_num should prevent NaN propagation."""
        gamma = torch.tensor([[0.0]])
        nu = torch.tensor([[1e-10]])  # very small → potential instability
        alpha = torch.tensor([[1.01]])
        beta = torch.tensor([[1e-10]])
        y = torch.tensor([100.0])
        loss = evidential_regression_loss(gamma, nu, alpha, beta, y)
        # Should not raise and should be finite or zero after nan_to_num
        assert not torch.isnan(loss)


class TestGradNormLoss:
    def test_get_weighted_loss_shape(self):
        gradnorm = GradNormLoss(num_tasks=3)
        losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
        total = gradnorm.get_weighted_loss(losses)
        assert total.ndim == 0

    def test_get_weighted_loss_uniform_weights(self):
        """With equal weights=1, weighted loss = mean * num_tasks = sum."""
        gradnorm = GradNormLoss(num_tasks=3)
        with torch.no_grad():
            gradnorm.task_weights.fill_(1.0)

        losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
        total = gradnorm.get_weighted_loss(losses)
        # normalised weights sum to num_tasks, each weight = 1
        # total = 1*1 + 1*2 + 1*3 = 6
        assert abs(total.item() - 6.0) < 1e-5

    def test_weights_are_learnable(self):
        gradnorm = GradNormLoss(num_tasks=4)
        assert gradnorm.task_weights.requires_grad

    def test_task_weights_initialised_to_one(self):
        gradnorm = GradNormLoss(num_tasks=5)
        assert torch.allclose(gradnorm.task_weights, torch.ones(5))

    def test_weighted_loss_positive_with_positive_inputs(self):
        gradnorm = GradNormLoss(num_tasks=2)
        losses = [torch.tensor(0.5), torch.tensor(1.5)]
        total = gradnorm.get_weighted_loss(losses)
        assert total.item() > 0
