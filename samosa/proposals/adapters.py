"""
Adaptive proposal strategies for MCMC sampling.
"""

import numpy as np
from samosa.core.state import ChainState
from samosa.core.proposal import ProposalBase, AdapterBase


class HaarioAdapter(AdapterBase):
    """
    Adaptive Metropolis algorithm (Haario et al., 2001).

    Adapts proposal covariance based on sample history using a
    recursive update formula with a fixed scaling factor.

    Reference:
        Haario, H., Saksman, E., & Tamminen, J. (2001).
        "An adaptive Metropolis algorithm." Bernoulli, 7(2), 223-242.

    Attributes:
        scale: Scaling factor for covariance (typically 2.38^2/d).
    """

    def __init__(
        self,
        scale: float,
        adapt_start: int = 500,
        adapt_end: int = 1000,
        eps: float = 1e-06,
    ) -> None:
        """
        Initialize Haario adapter.

        Args:
            scale: Covariance scaling factor (e.g., 2.38^2/dim for optimal mixing).
            adapt_start: Start adaptation at this iteration.
            adapt_end: Stop adaptation after this iteration.
            eps: Regularization constant for numerical stability.
        """
        super().__init__(adapt_start, adapt_end, eps)
        if scale <= 0:
            raise ValueError("Scale must be positive")
        self.scale = scale

    def adapt(self, proposal: ProposalBase, state: ChainState) -> None:
        """
        Adapt proposal using recursive covariance estimation.

        Updates both the empirical mean and covariance using all samples
        seen so far, with optional adaptation window.

        Args:
            proposal: Proposal to adapt.
            state: Current state with metadata containing 'iteration', 'mean', 'covariance'.

        Raises:
            ValueError: If state metadata is missing required fields.
        """
        if state.metadata is None:
            raise ValueError("State metadata is required for adaptation")

        # Extract state information
        x = (
            state.reference_position
            if state.reference_position is not None
            else state.position
        )
        iteration = state.metadata["iteration"]
        xmean_prev = state.metadata["mean"]
        xcov_prev = state.metadata["covariance"]
        dim = xmean_prev.shape[0]

        # Update empirical mean recursively
        xmean_new = xmean_prev + (x - xmean_prev) / iteration

        # Update empirical covariance (only during adaptation window)
        if iteration < self.adapt_start or iteration > self.adapt_end:
            xcov_new = xcov_prev
        else:
            # Recursive covariance update with scaling and regularization
            xcov_new = (
                iteration - 1
            ) / iteration * xcov_prev + self.scale / iteration * (
                iteration * (xmean_prev @ xmean_prev.T)
                - (iteration + 1) * (xmean_new @ xmean_new.T)
                + x @ x.T
                + self.eps * np.eye(dim)
            )

        # Update state metadata for next iteration
        state.metadata["mean"] = xmean_new
        state.metadata["covariance"] = xcov_new

        # Update proposal parameters
        proposal.update_parameters(mu=xmean_new, cov=xcov_new)


class GlobalAdapter(AdapterBase):
    """
    Global adaptive scaling algorithm (Andrieu & Thoms, 2008, Algorithm 4).

    Adapts both covariance and global scaling parameter based on
    acceptance rate to achieve a target acceptance rate.

    Reference:
        Andrieu, C., & Thoms, J. (2008). "A tutorial on adaptive MCMC."
        Statistics and Computing, 18(4), 343-373.

    Attributes:
        target_ar: Target acceptance rate (default: 0.234 for optimal mixing).
        C: Step size scaling constant.
        alpha: Decay exponent for diminishing adaptation.
    """

    def __init__(
        self,
        target_ar: float = 0.234,
        adapt_start: int = 500,
        adapt_end: int = 1000,
        C: float = 1.0,
        alpha: float = 0.5,
        eps: float = 1e-06,
    ) -> None:
        """
        Initialize global adapter.

        Args:
            target_ar: Target acceptance rate (typically 0.234 in high dimensions, 0.44 for low dimensions).
            adapt_start: Start adaptation at this iteration.
            adapt_end: Stop adaptation after this iteration.
            C: Constant for step size gamma = C / n^alpha.
            alpha: Exponent for diminishing adaptation (0.5 < alpha <= 1).
            eps: Regularization constant for numerical stability.
        """
        super().__init__(adapt_start, adapt_end, eps)
        if not (0 < target_ar < 1):
            raise ValueError("Target acceptance rate must be in (0, 1)")
        if C <= 0:
            raise ValueError("C must be positive")
        if not (0 < alpha <= 1):
            raise ValueError("Alpha must be in (0, 1]")

        self.target_ar = target_ar
        self.C = C
        self.alpha = alpha

    def adapt(self, proposal: ProposalBase, state: ChainState) -> None:
        """
        Adapt proposal using global scaling and acceptance rate.

        Updates empirical mean, covariance, and global scaling parameter
        using diminishing adaptation with step size gamma_n = C / n^alpha.

        Args:
            proposal: Proposal to adapt.
            state: Current state with metadata containing 'iteration', 'mean',
                   'covariance', 'lambda', 'acceptance_probability'.

        Raises:
            ValueError: If state metadata is missing required fields.
        """
        if state.metadata is None:
            raise ValueError("State metadata is required for adaptation")

        # Extract state information
        x = (
            state.reference_position
            if state.reference_position is not None
            else state.position
        )
        iteration = state.metadata["iteration"]
        xmean_prev = state.metadata["mean"]
        xcov_prev = state.metadata["covariance"]
        lambda_prev = state.metadata["lambda"]
        acceptance_prob = state.metadata["acceptance_probability"]
        dim = xmean_prev.shape[0]

        # Compute diminishing step size
        gamma = self.C / (iteration**self.alpha)

        # Update empirical mean with step size gamma
        xmean_new = xmean_prev + gamma * (x - xmean_prev)

        # Update covariance and scaling (only during adaptation window)
        if iteration < self.adapt_start or iteration > self.adapt_end:
            xcov_new = xcov_prev
            lambda_new = lambda_prev
        else:
            # Update empirical covariance
            xcov_new = xcov_prev + gamma * (
                (x - xmean_prev) @ (x - xmean_prev).T
                - xcov_prev
                + self.eps * np.eye(dim)
            )

            # Update global scaling based on acceptance rate
            lambda_new = lambda_prev * np.exp(
                gamma * (acceptance_prob - self.target_ar)
            )

        # Update state metadata for next iteration
        state.metadata["mean"] = xmean_new
        state.metadata["covariance"] = xcov_new
        state.metadata["lambda"] = lambda_new

        # Update proposal with scaled covariance
        proposal.update_parameters(mu=xmean_new, cov=lambda_new * xcov_new)
