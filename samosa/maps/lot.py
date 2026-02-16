"""
Linear Optimal Transport (LOT) map implementation.

This module provides a linear transport map that optimally couples two
Gaussian distributions using the closed-form solution for linear optimal
transport between Gaussians. This map is particularly useful for multi-fidelity
MCMC where you need to couple coarse and fine chains.
"""

# Imports
import numpy as np
from scipy.linalg import sqrtm

from samosa.core.map import TransportMap
from samosa.core.state import ChainState
from samosa.core.model import ModelProtocol
from samosa.utils.post_processing import get_position_from_states
from typing import List, Optional, Tuple


class LinearOptimalTransportMap(TransportMap):
    """
    Linear Optimal Transport map for coupling Gaussian distributions.

    This map computes the optimal linear transformation between two Gaussian
    distributions (typically fine and coarse chains in multi-fidelity MCMC).
    The map is bijective and provides forward/inverse transforms with exact
    Jacobian determinants.

    The map is adapted using sample statistics from both fine and coarse chains,
    computing the optimal linear transport that minimizes the Wasserstein-2
    distance between the two distributions.
    """

    def __init__(
        self,
        dim: int,
        adapt_start: int = 500,
        adapt_end: int = 1000,
        adapt_interval: int = 100,
        reference_model: Optional[ModelProtocol] = None,
        eps: float = 1e-6,
    ) -> None:
        """
        Initialize the linear optimal transport map.

        Args:
            dim: Dimension of the parameter space.
            adapt_start: Iteration to start adaptation.
            adapt_end: Iteration to end adaptation.
            adapt_interval: Adapt every ``adapt_interval`` iterations.
            reference_model: Optional reference model. If ``None``, assumes
                standard Gaussian for the coarse/reference distribution.
            eps: Regularization parameter added to covariance matrices for
                numerical stability.
        """
        super().__init__(
            dim=dim,
            adapt_start=adapt_start,
            adapt_end=adapt_end,
            adapt_interval=adapt_interval,
        )

        self.reference_model = reference_model
        self.A = np.eye(dim)
        self.b = np.zeros((dim, 1))
        self.eps = eps

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward map: transform from target space to reference space.

        Args:
            x: Position in target space, shape ``(dim, N)`` or ``(dim, 1)``.

        Returns:
            Tuple of (reference_position, log_determinant):
                - reference_position: shape ``(dim, N)`` or ``(dim, 1)``
                - log_determinant: log |det(A)|
        """
        # Evaluate the linear map: r = A @ x + b
        r = self.A @ x + self.b

        # Compute log determinant of the linear transformation
        log_det = np.log(np.abs(np.linalg.det(self.A)))

        return r, log_det

    def inverse(self, r: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Inverse map: transform from reference space back to target space.

        Args:
            r: Position in reference space, shape ``(dim, N)`` or ``(dim, 1)``.

        Returns:
            Tuple of (position, log_determinant):
                - position: shape ``(dim, N)`` or ``(dim, 1)``
                - log_determinant: log |det(A^{-1})| = -log |det(A)|
        """
        # Evaluate the inverse linear map: x = A^{-1} @ (r - b)
        x = np.linalg.inv(self.A) @ (r - self.b)

        # Log determinant of inverse is negative of forward log determinant
        log_det = np.log(np.abs(np.linalg.det(np.linalg.inv(self.A))))

        return x, log_det

    def adapt(
        self,
        samples: List[ChainState],
        force_adapt: bool = False,
        paired_samples: Optional[List[ChainState]] = None,
    ) -> None:
        """
        Adapt the map using fine and coarse samples.

        This map requires both fine samples (``samples``) and coarse samples
        (``paired_samples``) to compute the optimal linear transport between
        their empirical distributions.

        Args:
            samples: Fine samples (target distribution).
            force_adapt: If ``True``, bypass adaptation window/interval checks.
            paired_samples: Coarse samples (reference distribution). Required
                for this map type. If ``None`` and ``reference_model`` is also
                ``None``, uses standard Gaussian as reference.
        """
        if paired_samples is None:
            raise ValueError(
                "LinearOptimalTransportMap.adapt() requires paired_samples "
                "(coarse samples) to compute the optimal transport map."
            )

        if not self._should_adapt(samples, force_adapt=force_adapt):
            return None

        iteration = self._extract_iteration(samples)
        print(f"Adapting Linear Optimal Transport map at iteration {iteration}")

        # Get positions from states
        positions_fine = get_position_from_states(samples)
        positions_coarse = get_position_from_states(paired_samples)

        # Compute empirical statistics for fine distribution
        self.mu_fine = np.mean(positions_fine, axis=1, keepdims=True)
        self.cov_fine = np.cov(positions_fine) + self.eps * np.eye(self.dim)

        # Compute empirical statistics for coarse/reference distribution
        if self.reference_model is None:
            # Use standard Gaussian as reference
            self.mu_coarse = np.zeros((self.dim, 1))
            self.cov_coarse = np.eye(self.dim)
        else:
            # Use empirical statistics from coarse samples
            self.mu_coarse = np.mean(positions_coarse, axis=1, keepdims=True)
            self.cov_coarse = np.cov(positions_coarse) + self.eps * np.eye(self.dim)

        # Compute the optimal linear transport map (Fine -> Coarse)
        # Using the closed-form solution for linear optimal transport between Gaussians:
        # A = sqrt(C_coarse) @ inv(sqrt(sqrt(C_coarse) @ C_fine @ sqrt(C_coarse))) @ sqrt(C_coarse)
        # b = mu_coarse - A @ mu_fine
        sqrt_cov_coarse = sqrtm(self.cov_coarse)
        inv_sqrt = np.linalg.inv(
            sqrtm(sqrt_cov_coarse @ self.cov_fine @ sqrt_cov_coarse)
        )
        self.A = sqrt_cov_coarse @ inv_sqrt @ sqrt_cov_coarse
        self.b = self.mu_coarse - self.A @ self.mu_fine
