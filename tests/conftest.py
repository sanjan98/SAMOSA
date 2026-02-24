"""
Shared fixtures for SAMOSA tests.

Provides unseen-posterior models (Gaussian direct, Gaussian component-based,
optional banana-like) for single-chain and kernel tests.
"""

import numpy as np
import pytest


# --------------------------------------------------
# Unseen posterior: Gaussian (direct log_posterior)
# --------------------------------------------------


class GaussianPosterior:
    """Unseen posterior: log_posterior = -0.5 * sum((x - mu)^2). Shape (d, 1)."""

    def __init__(self, dim: int = 2, mu: np.ndarray | None = None):
        self.dim = dim
        self.mu = mu if mu is not None else np.zeros((dim, 1))

    def __call__(self, params: np.ndarray) -> dict:
        x = np.asarray(params).reshape(-1, 1)
        diff = x - self.mu
        log_posterior = float(-0.5 * np.sum(diff**2))
        return {"log_posterior": log_posterior}


@pytest.fixture
def gaussian_posterior_2d():
    """Gaussian unseen posterior, dim=2, mu=0."""
    return GaussianPosterior(dim=2)


@pytest.fixture
def gaussian_posterior_3d():
    """Gaussian unseen posterior, dim=3, mu=0."""
    return GaussianPosterior(dim=3)


@pytest.fixture
def gaussian_posterior_offset():
    """Gaussian unseen posterior, dim=2, mu=[1, -0.5]."""
    return GaussianPosterior(dim=2, mu=np.array([[1.0], [-0.5]]))


# --------------------------------------------------
# Unseen posterior: Gaussian (component-based)
# --------------------------------------------------


class GaussianPosteriorComponents:
    """Same Gaussian as log_prior + log_likelihood so ChainState auto-sets log_posterior."""

    def __init__(self, dim: int = 2, mu: np.ndarray | None = None):
        self.dim = dim
        self.mu = mu if mu is not None else np.zeros((dim, 1))

    def __call__(self, params: np.ndarray) -> dict:
        x = np.asarray(params).reshape(-1, 1)
        diff = x - self.mu
        log_prior = float(-0.5 * np.sum(diff**2) * 0.5)
        log_likelihood = float(-0.5 * np.sum(diff**2) * 0.5)
        return {"log_prior": log_prior, "log_likelihood": log_likelihood}


@pytest.fixture
def gaussian_posterior_components_2d():
    """Component-based Gaussian unseen posterior, dim=2."""
    return GaussianPosteriorComponents(dim=2)


# --------------------------------------------------
# Unseen posterior: banana-like (non-Gaussian)
# --------------------------------------------------


class BananaLikePosterior:
    """Simple non-Gaussian: banana-style or heavy-tail, log_posterior only."""

    def __init__(self, dim: int = 2):
        self.dim = dim

    def __call__(self, params: np.ndarray) -> dict:
        x = np.asarray(params).reshape(-1, 1)
        # Banana-style: x1^2 + (b*x1^2 - x2)^2 style, simplified
        x1 = float(x[0, 0])
        x2 = float(x[1, 0]) if x.shape[0] >= 2 else 0.0
        if self.dim >= 2:
            b = 0.1
            log_posterior = float(-(x1**2) - (b * x1**2 - x2) ** 2)
        else:
            log_posterior = float(-0.5 * np.sum(x**2))
        return {"log_posterior": log_posterior}


@pytest.fixture
def banana_like_posterior():
    """Banana-like unseen posterior, dim=2."""
    return BananaLikePosterior(dim=2)
