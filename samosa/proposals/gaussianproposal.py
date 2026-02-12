"""
Gaussian-based proposal distributions for MCMC sampling.
"""

from typing import Optional

import numpy as np

from samosa.core.proposal import ProposalBase
from samosa.core.state import ChainState
from samosa.utils.tools import lognormpdf, sample_multivariate_gaussian


class GaussianRandomWalk(ProposalBase):
    """
    Gaussian random walk proposal: q(x'|x) = N(x' | x, Σ).
    
    Proposes new states by adding a x-mean Gaussian increment
    to the current state. If x is zero, this is a symmetric random walk.
    """

    def __init__(self, mu: np.ndarray, cov: np.ndarray) -> None:
        """
        Initialize Gaussian random walk proposal.
        
        Args:
            mu: Mean vector (d, 1) - typically zero for random walk.
            cov: Covariance matrix (d, d) for the increment.
            
        """
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("Covariance must be a square matrix")
        
        dim = cov.shape[0]
        if mu.shape[0] != dim:
            raise ValueError(f"Mean dimension {mu.shape[0]} does not match covariance dimension {dim}")
        
        super().__init__(mu, cov)

    def sample(self, current_state: ChainState, common_step: Optional[np.ndarray] = None) -> ChainState:
        """
        Generate candidate by adding Gaussian increment to current position.
        
        Args:
            current_state: Current state.
            common_step: Optional standard normal random variable for coupling.
            
        Returns:
            Proposed state with position = current + N(0, Σ).
        """
        if common_step is None:
            step = sample_multivariate_gaussian(self.mu, self.cov)
        else:
            step = self.mu + np.linalg.cholesky(self.cov) @ common_step
        
        return ChainState(position=current_state.position + step)

    def proposal_logpdf(self, current_state: ChainState, proposed_state: ChainState) -> tuple[float, float]:
        """
        Compute log proposal densities (symmetric for zero mean random walk).
        
        Args:
            current_state: Current state.
            proposed_state: Proposed state.
            
        Returns:
            (log q(proposed|current), log q(current|proposed))
        """
        logq_forward = lognormpdf(proposed_state.position, current_state.position + self.mu, self.cov)
        logq_reverse = lognormpdf(current_state.position, proposed_state.position + self.mu, self.cov)
        
        return float(logq_forward), float(logq_reverse)
    
    def update_parameters(self, mu: Optional[np.ndarray] = None, cov: Optional[np.ndarray] = None) -> None:
        """
        Update parameters (ignores mu for random walk, only updates covariance).
        
        Args:
            mu: Ignored for random walk proposals.
            cov: New covariance matrix (optional).
        """
        if cov is not None:
            self.cov = cov


class IndependentProposal(ProposalBase):
    """
    Independent Gaussian proposal: q(x'|x) = N(x' | μ, Σ).
    
    Proposes new states from a fixed Gaussian distribution, independent
    of the current state. 
    """

    def __init__(self, mu: np.ndarray, cov: np.ndarray) -> None:
        """
        Initialize independent Gaussian proposal.
        
        Args:
            mu: Mean vector (d,) or (d, 1) of the proposal distribution.
            cov: Covariance matrix (d, d) of the proposal distribution.
        """
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("Covariance must be a square matrix")
        if mu.shape[0] != cov.shape[0]:
            raise ValueError(f"Mean dimension {mu.shape[0]} does not match covariance dimension {cov.shape[0]}")
        
        super().__init__(mu, cov)

    def sample(self, current_state: ChainState, common_step: Optional[np.ndarray] = None) -> ChainState:
        """
        Generate candidate from independent Gaussian distribution.
        
        Args:
            current_state: Current state (not used for independent proposal).
            common_step: Optional standard normal random variable for coupling.
            
        Returns:
            Proposed state with position ~ N(μ, Σ).
        """
        if common_step is None:
            position = sample_multivariate_gaussian(self.mu, self.cov)
        else:
            position = self.mu + np.linalg.cholesky(self.cov) @ common_step
        
        return ChainState(position=position)

    def proposal_logpdf(self, current_state: ChainState, proposed_state: ChainState) -> tuple[float, float]:
        """
        Compute log proposal densities (independent of current state).
        
        Args:
            current_state: Current state.
            proposed_state: Proposed state.
            
        Returns:
            (log q(proposed), log q(current))
        """
        logq_forward = lognormpdf(proposed_state.position, self.mu, self.cov)
        logq_reverse = lognormpdf(current_state.position, self.mu, self.cov)
        
        return float(logq_forward), float(logq_reverse)


class PreCrankedNicholson(ProposalBase):
    """
    Preconditioned Crank-Nicolson (pCN) proposal.
    
    A gradient-free proposal designed for sampling in infinite dimensions,
    but also effective in high-dimensional finite spaces. The proposal has
    form: x' = √(1-β²)(x-μ) + μ + β·z where z ~ N(0, Σ).
    
    The parameter β ∈ (0,1] controls exploration vs exploitation and can
    be adapted based on acceptance rate.
    
    Reference:
        Cotter, S. L., et al. (2013). "MCMC methods for functions: 
        modifying old algorithms to make them faster." Statistical Science, 28(3).
    
    Attributes:
        beta: Step size parameter in (0, 1].
        target_acceptance: Desired acceptance rate for beta adaptation.
    """

    def __init__(
        self, 
        mu: np.ndarray, 
        cov: np.ndarray, 
        beta: float, 
        target_acceptance: float = 0.25, 
        long_alpha: float = 0.99, 
        short_alpha: float = 0.9, 
        adjust_rate: float = 0.01, 
        beta_min: float = 1e-3, 
        beta_max: float = 1 - 1e-3
    ) -> None:
        """
        Initialize pCN proposal.
        
        Args:
            mu: Reference/mean position (d,) or (d, 1).
            cov: Covariance matrix (d, d).
            beta: Initial step size in (0, 1].
            target_acceptance: Target acceptance rate for adaptation (default: 0.25).
            long_alpha: Smoothing factor for long-term average (default: 0.99).
            short_alpha: Smoothing factor for short-term average (default: 0.9).
            adjust_rate: Rate of beta adjustment (default: 0.01).
            beta_min: Minimum allowed beta value (default: 1e-3).
            beta_max: Maximum allowed beta value (default: 1-1e-3).
            
        Raises:
            ValueError: If beta not in (0, 1] or other parameters invalid.
        """
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("Covariance must be a square matrix")
        if mu.shape[0] != cov.shape[0]:
            raise ValueError(f"Mean dimension {mu.shape[0]} does not match covariance dimension {cov.shape[0]}")
        if not (0 < beta <= 1):
            raise ValueError("Beta must be in (0, 1]")
        if not (0 < target_acceptance < 1):
            raise ValueError("Target acceptance must be in (0, 1)")
        if not (0 < beta_min < beta_max < 1):
            raise ValueError("Must have 0 < beta_min < beta_max < 1")
        
        super().__init__(mu, cov)
        
        self.beta = beta
        self.target_acceptance = target_acceptance
        self.long_alpha = long_alpha
        self.short_alpha = short_alpha
        self.adjust_rate = adjust_rate
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        # Running averages for adaptive beta
        self.long_average = target_acceptance
        self.short_average = target_acceptance

    def sample(self, current_state: ChainState, common_step: Optional[np.ndarray] = None) -> ChainState:
        """
        Generate pCN proposal: x' = √(1-β²)(x-μ) + μ + β·z.
        
        Args:
            current_state: Current state.
            common_step: Optional standard normal random variable for coupling.
            
        Returns:
            Proposed state from pCN distribution.
        """
        dim = current_state.position.shape[0]
        
        if common_step is None:
            z = sample_multivariate_gaussian(np.zeros((dim, 1)), self.cov)
        else:
            z = np.linalg.cholesky(self.cov) @ common_step
        
        # pCN formula: x' = sqrt(1-β²)(x-μ) + μ + β·z
        proposal = (
            np.sqrt(1 - self.beta**2) * (current_state.position - self.mu)
            + self.mu
            + self.beta * z
        )
        
        return ChainState(position=proposal)

    def proposal_logpdf(self, current_state: ChainState, proposed_state: ChainState) -> tuple[float, float]:
        """
        Compute log pCN proposal densities.
        
        Args:
            current_state: Current state.
            proposed_state: Proposed state.
            
        Returns:
            (log q(proposed|current), log q(current|proposed))
        """
        # Forward: proposed = sqrt(1-β²)(current-μ) + μ + β·z
        mean_forward = np.sqrt(1 - self.beta**2) * (current_state.position - self.mu) + self.mu
        
        # Reverse: current = sqrt(1-β²)(proposed-μ) + μ + β·z
        mean_reverse = np.sqrt(1 - self.beta**2) * (proposed_state.position - self.mu) + self.mu
        
        # Variance is β²Σ in both directions
        scaled_cov = (self.beta**2) * self.cov
        
        logq_forward = lognormpdf(proposed_state.position, mean_forward, scaled_cov)
        logq_reverse = lognormpdf(current_state.position, mean_reverse, scaled_cov)
        
        return float(logq_forward), float(logq_reverse)
    
    def update_parameters(self, mu: Optional[np.ndarray] = None, cov: Optional[np.ndarray] = None) -> None:
        """
        Update parameters (ignores mu for pCN, only updates covariance).
        
        Args:
            mu: Ignored for pCN proposals.
            cov: New covariance matrix (optional).
        """
        if cov is not None:
            self.cov = cov

    def adapt(self, state: ChainState) -> None:
        """
        Adapt beta parameter based on acceptance rate.
        
        Uses dual averaging with long-term and short-term running averages
        to adjust beta toward achieving the target acceptance rate.
        
        Args:
            state: Current state with 'is_accepted' in metadata.
        """
        is_accepted = state.metadata.get('is_accepted', False) if state.metadata is not None else False
        
        # Update running averages
        self.long_average = self.long_alpha * self.long_average + is_accepted * (1 - self.long_alpha)
        self.short_average = self.short_alpha * self.short_average + is_accepted * (1 - self.short_alpha)
        
        # Adjust beta using both averages (dual averaging)
        adjusted_beta = (
            self.beta
            + self.adjust_rate * (self.long_average - self.target_acceptance)
            + 0.5 * self.adjust_rate * (self.short_average - self.target_acceptance)
        )
        
        # Clamp to valid range
        self.beta = float(np.clip(adjusted_beta, self.beta_min, self.beta_max))