"""
Class file for the proposal adapters
"""

# Imports
import numpy as np
from typing import Tuple
from samosa.core.state import ChainState
from samosa.core.proposal import ProposalProtocol, AdapterBase
from samosa.proposals.gaussianproposal import GaussianRandomWalk, IndependentProposal

# Concrete Adapters
class HaarioAdapter(AdapterBase):
    """Haario et al. adaptation strategy"""
    
    def __init__(self, scale: float, adapt_start: int = 500, adapt_end: int = 1000, eps: float = 1e-06):
        super().__init__(adapt_start, adapt_end, eps)
        self.scale = scale
    
    def adapt(self, proposal: ProposalProtocol, state: ChainState) -> None:
        """Adapt the proposal covariance based on the history of samples"""

        x = state.reference_position if state.reference_position is not None else state.position
        iteration = state.metadata['iteration']
        xmean_minus = state.metadata['mean']
        xcov_minus = state.metadata['covariance']
        dim = xmean_minus.shape[0]

        # Update the mean
        xmean = xmean_minus + (x - xmean_minus) / iteration

        # Check if adaptation is needed
        if iteration < self.adapt_start or iteration > self.adapt_end:
            xcov = xcov_minus
        else:
            # Update the covariance
            xcov = (iteration - 1) / iteration * xcov_minus + self.scale / iteration * (iteration * xmean_minus @ xmean_minus.T - (iteration + 1) * xmean @ xmean.T + x @ x.T + self.eps * np.eye(dim))

        # Update the state metadata
        state.metadata['mean'] = xmean
        state.metadata['covariance'] = xcov

        # Update the proposal parameters
        if isinstance(proposal, GaussianRandomWalk):
            proposal.mu = np.zeros((dim, 1))
            proposal.cov = xcov
        elif isinstance(proposal, IndependentProposal):
            proposal.mu = xmean
            proposal.cov = xcov
        # Add other proposal types as needed ...

class GlobalAdapter(AdapterBase):
    """Andrieu et al. adaptation strategy (Algorithm 4)"""
    
    def __init__(self, ar: float = 0.234, adapt_start: int = 500, adapt_end: int = 1000, C: float = 1.0, alpha: float = 0.5, eps: float = 1e-06):
        super().__init__(adapt_start, adapt_end, eps)
        self.ar = ar
        self.C = C
        self.alpha = alpha
        
    def adapt(self, proposal: ProposalProtocol, state: ChainState,) -> None:
        """Adapt the proposal covariance based on the target acceptance rate"""

        x = state.reference_position if state.reference_position is not None else state.position
        iteration = state.metadata['iteration']
        gamma = (self.C) / (iteration ** self.alpha)
        xmean_minus = state.metadata['mean']
        xcov_minus = state.metadata['covariance']
        lambda__minus = state.metadata['lambda']
        ar = state.metadata['acceptance_probability']
        dim = xmean_minus.shape[0]

        # Update the mean
        xmean = xmean_minus + gamma * (x - xmean_minus)

        # Check if adaptation is needed
        if iteration < self.adapt_start or iteration > self.adapt_end:
            xcov = xcov_minus
            lambda_ = lambda__minus
        else:
            # Update the covariance
            xcov = xcov_minus + gamma * ((x - xmean_minus) @ (x - xmean_minus).T - xcov_minus + self.eps * np.eye(dim))
            lambda_ = lambda__minus * np.exp(gamma * (ar - self.ar))
        
        # Update the state metadata
        state.metadata['mean'] = xmean
        state.metadata['covariance'] = xcov
        state.metadata['lambda'] = lambda_
        
        # Update the proposal parameters
        if isinstance(proposal, GaussianRandomWalk):
            proposal.mu = np.zeros((dim, 1))
            proposal.cov = lambda_ * xcov
        elif isinstance(proposal, IndependentProposal):
            proposal.mu = xmean
            proposal.cov = lambda_ * xcov
        # Add other proposal types as needed ...
