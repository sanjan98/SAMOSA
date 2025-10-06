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

    def sample(self, _: ChainState, common_step: Optional[np.ndarray]) -> ChainState:
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

    def __init__(self, mu: np.ndarray, sigma: np.ndarray, beta: float):
        self.mu = mu
        self.sigma = sigma
        self.cov = sigma.copy()
        assert 0 < beta <= 1, "Beta must be in (0,1)"
        self.beta = beta

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

    def adapt(self, _: ChainState):
        """No adaptation for this proposal"""
        pass
