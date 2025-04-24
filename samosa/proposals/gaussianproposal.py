"""
Gaussian-based proposals for MCMC sampling
"""

import numpy as np
from typing import Tuple
from samosa.core.proposal import ProposalProtocol
from samosa.core.state import ChainState
from samosa.utils.tools import sample_multivariate_gaussian, lognormpdf

class GaussianRandomWalk(ProposalProtocol):
    """Random walk proposal centered at current state"""

    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma
        self.cov = sigma.copy() 
    
    def sample(self, current_state: ChainState) -> ChainState:
        step = sample_multivariate_gaussian(self.mu, self.cov)
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
    
    def sample(self, _: ChainState) -> ChainState:
        return ChainState(
            position=sample_multivariate_gaussian(self.mu, self.cov)
        )
    
    def proposal_logpdf(self, current_state: ChainState, proposed_state: ChainState) -> tuple[float, float]:
        """Calculate forward and reverse log probability"""
        logq_forward = lognormpdf(proposed_state.position, self.mu, self.cov)
        logq_reverse = lognormpdf(current_state.position, self.mu, self.cov)
        return logq_forward, logq_reverse
    
    def adapt(self, _: ChainState):
        """No adaptation for this proposal"""
        pass
