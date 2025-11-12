"""
Gaussian-based proposals for MCMC sampling
"""

from typing import Tuple, Optional

import numpy as np

from samosa.core.proposal import ProposalProtocol
from samosa.core.state import ChainState
from samosa.utils.tools import lognormpdf, sample_multivariate_gaussian


class GaussianRandomWalk(ProposalProtocol):
    """Random walk proposal centered at current state"""

    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma
        self.cov = sigma.copy()

    def sample(self, current_state: ChainState, common_step: Optional[np.ndarray] = None) -> ChainState:
        """Generate candidate state from current state"""
        if common_step is None:
            step = sample_multivariate_gaussian(self.mu, self.cov)
        else:
            step = self.mu + np.linalg.cholesky(self.cov) @ common_step
        return ChainState(position=current_state.position + step)

    def proposal_logpdf(self, current_state: ChainState, proposed_state: ChainState) -> Tuple[float, float]:
        """Calculate forward and reverse log probability"""
        logq_forward = lognormpdf(proposed_state.position, current_state.position + self.mu, self.cov)
        logq_reverse = lognormpdf(current_state.position, proposed_state.position + self.mu, self.cov)

        return logq_forward, logq_reverse

    def adapt(self, _: ChainState):
        """No adaptation for this proposal"""
        pass

class IndependentProposal(ProposalProtocol):
    """Independent proposal from fixed Gaussian distribution"""

    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma
        self.cov = sigma.copy()

    def sample(self, _: ChainState, common_step: Optional[np.ndarray] = None) -> ChainState:
        if common_step is None:
            return ChainState(position=sample_multivariate_gaussian(self.mu, self.cov))
        else:
            return ChainState(position=self.mu + np.linalg.cholesky(self.cov) @ common_step)

    def proposal_logpdf(self, current_state: ChainState, proposed_state: ChainState) -> tuple[float, float]:
        """Calculate forward and reverse log probability"""
        logq_forward = lognormpdf(proposed_state.position, self.mu, self.cov)
        logq_reverse = lognormpdf(current_state.position, self.mu, self.cov)
        return logq_forward, logq_reverse

    def adapt(self, _: ChainState):
        """No adaptation for this proposal"""
        pass

class PreCrankedNicholson(ProposalProtocol):
    """Preconditioned Cranked-Nicholson proposal"""

    def __init__(self, mu: np.ndarray, sigma: np.ndarray, beta: float, target_acceptance: float = 0.25, long_alpha: float = 0.99, short_alpha: float = 0.9, adjust_rate: float = 0.01, beta_min: float = 1e-3, beta_max: float = 1-1e-3, eps: float = 1e-06):
        self.mu = mu
        self.sigma = sigma
        self.cov = sigma.copy()
        assert 0 < beta <= 1, "Beta must be in (0,1)"
        self.beta = beta

        # --- Adaptive beta parameters ---
        self.target_acceptance = target_acceptance
        self.long_alpha = long_alpha
        self.short_alpha = short_alpha
        self.adjust_rate = adjust_rate
        self.beta_min = beta_min
        self.beta_max = beta_max
        # --- Running averages ---
        self.long_average = target_acceptance
        self.short_average = target_acceptance

        # --- Adaptation parameters for covariance ---
        self.eps = eps
        self.scale = 2.4**2 / mu.shape[0]


    def sample(self, current_state: ChainState, common_step: Optional[np.ndarray] = None) -> ChainState:
        dim = current_state.position.shape[0]
        if common_step is None:
            z = sample_multivariate_gaussian(np.zeros((dim, 1)), self.cov)
        else:
            z = common_step
        proposal = np.sqrt(1 - self.beta**2) * (current_state.position - self.mu) + self.mu + self.beta * z
        return ChainState(position=proposal)

    def proposal_logpdf(self, current_state: ChainState, proposed_state: ChainState) -> tuple[float, float]:
        """Calculate forward and reverse log probability"""
        dim = current_state.position.shape[0]
        mean_forward = np.sqrt(1 - self.beta**2) * (current_state.position - self.mu) + self.mu
        mean_reverse = np.sqrt(1 - self.beta**2) * (proposed_state.position - self.mu) + self.mu

        logq_forward = lognormpdf(proposed_state.position, mean_forward, (self.beta**2) * self.cov)
        logq_reverse = lognormpdf(current_state.position, mean_reverse, (self.beta**2) * self.cov)

        return logq_forward, logq_reverse

    def adapt(self, state: ChainState):
        """Adapt the beta parameter based on acceptance rate"""
        is_accepted = state.metadata.get('is_accepted', False)
        self.long_average = self.long_alpha * self.long_average + is_accepted * (1 - self.long_alpha)
        self.short_average = self.short_alpha * self.long_average + is_accepted * (1 - self.short_alpha)
        # Adjust beta using both averages
        adjusted_beta = (
            self.beta +
            self.adjust_rate * (self.long_average - self.target_acceptance) +
            0.5 * self.adjust_rate * (self.short_average - self.target_acceptance)
        )
        # Clamp beta within the preset range
        self.beta = np.clip(adjusted_beta, self.beta_min, self.beta_max)

        # Doing this to get the updated mean and covariance for resynchronization
        x = state.reference_position if state.reference_position is not None else state.position
        iteration = state.metadata['iteration']
        xmean_minus = state.metadata['mean']
        xcov_minus = state.metadata['covariance']
        dim = xmean_minus.shape[0]

        # Update the mean
        xmean = xmean_minus + (x - xmean_minus) / iteration
        xcov = (iteration - 1) / iteration * xcov_minus + self.scale / iteration * (iteration * xmean_minus @ xmean_minus.T - (iteration + 1) * xmean @ xmean.T + x @ x.T + self.eps * np.eye(dim))

        # Update the state metadata
        state.metadata['mean'] = xmean
        state.metadata['covariance'] = xcov
        
