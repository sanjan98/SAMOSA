"""
Gaussian-based proposals for MCMC sampling
"""

from abc import ABC
import numpy as np
from core.proposal import Proposal
from core.state import ChainState
import mcmc.utils.mcmc_utils_and_plot as utils

class GaussianProposal(Proposal, ABC):
    """Base class for Gaussian-based proposals"""
    
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def proposal_logpdf(self, current_state: ChainState, proposed_state: ChainState) -> tuple[float, float]:
        """Shared logpdf calculation for Gaussian proposals"""
        logq_forward = utils.lognormpdf(proposed_state.position, current_state.position + self.mu, self.sigma)
        logq_reverse = utils.lognormpdf(current_state.position, proposed_state.position + self.mu, self.sigma)
        
        return logq_forward, logq_reverse

class GaussianRandomWalk(GaussianProposal):
    """Random walk proposal centered at current state"""
    
    def sample(self, current_state: ChainState) -> ChainState:
        step = utils.sample_multivariate_gaussian(self.mu, self.sigma)
        return ChainState(position=current_state.position + step)

class IndependentProposal(GaussianProposal):
    """Independent proposal from fixed Gaussian distribution"""
    
    def sample(self, _) -> ChainState:
        return ChainState(
            position=utils.sample_multivariate_gaussian(self.mu, self.sigma)
        )
    
    def proposal_logpdf(self, current_state: ChainState, proposed_state: ChainState) -> tuple[float, float]:
        """Override for independent case (current_state irrelevant)"""
        logq_forward = utils.lognormpdf(proposed_state.position, self.mu, self.sigma)
        logq_reverse = utils.lognormpdf(current_state.position, self.mu, self.sigma)
        return logq_forward, logq_reverse
    
class HaarioAdaptiveProposal(GaussianProposal):
    """Adaptive proposal using Haario et al. method"""
    
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, scale: float, adapt_start: int = 500, adapt_end: int = 1000, eps: float = 1e-06):
        super().__init__(mu, sigma)
        self.scale = scale
        self.adapt_start = adapt_start
        self.adapt_end = adapt_end
        self.eps = eps
        self.dim = mu.shape[0]

        self.cov = sigma.copy()

    def sample(self, current_state: ChainState) -> ChainState:
        step = utils.sample_multivariate_gaussian(self.mu, self.cov)
        return ChainState(position=current_state.position + step)
    
    def adapt(self, state: ChainState):

        """Adapt the proposal covariance based on the history of samples"""
        x = state.position
        iteration = state.metadata['iteration']
        xmean_minus = state.metadata['mean']
        xcov_minus = state.metadata['covariance']

        # Update the mean
        xmean = xmean_minus + (x - xmean_minus) / iteration

        # Check if adaptation is needed
        if iteration < self.adapt_start or iteration > self.adapt_end:
            xcov = xcov_minus
        else:
            # Update the covariance
            xcov = (iteration - 1) / iteration * xcov_minus + self.scale / iteration * (iteration * xmean_minus @ xmean_minus.T - (iteration + 1)* xmean @ xmean.T + x @ x.T + self.eps * np.eye(self.dim))

        # Update the state metadata
        state.metadata['mean'] = xmean
        state.metadata['covariance'] = xcov

        # Update the proposal parameters
        self.mu = np.zeros((self.dim, 1))
        self.cov = xcov

class GlobalAdaptiveProposal(GaussianProposal):
    """Adaptive proposal using Andrieu et al. method (Algorithm 4)"""
    
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, ar: float = 0.234, adapt_start: int = 500, adapt_end: int = 1000, C: float = 1.0, alpha: float = 0.5, eps: float = 1e-06):
        super().__init__(mu, sigma)
        self.ar = ar
        self.adapt_start = adapt_start
        self.adapt_end = adapt_end
        self.C = C
        self.alpha = alpha
        self.eps = eps
        self.dim = mu.shape[0]

        self.cov = sigma.copy()

    def sample(self, current_state: ChainState) -> ChainState:
        step = utils.sample_multivariate_gaussian(self.mu, self.cov)
        return ChainState(position=current_state.position + step)
    
    def adapt(self, state: ChainState):
        """Adapt the proposal covariance based on the target acceptance rate"""

        x = state.position
        iteration = state.metadata['iteration']
        gamma = self.C / iteration ** self.alpha
        xmean_minus = state.metadata['mean']
        xcov_minus = state.metadata['covariance']
        lambda__minus = state.metadata['lambda']
        ar = state.metadata['acceptance_rate']

        # Update the mean
        xmean = xmean_minus + gamma * (x - xmean_minus)

        # Check if adaptation is needed
        if iteration < self.adapt_start or iteration > self.adapt_end:
            xcov = xcov_minus
            lambda_ = lambda__minus
        else:
            # Update the covariance
            xcov = xcov_minus + gamma * ((x - xmean_minus) @ (x - xmean_minus).T - xcov_minus + self.eps * np.eye(self.dim))
            lambda_ = lambda__minus * np.exp(ar - self.ar)
        
        # Update the state metadata
        state.metadata['mean'] = xmean
        state.metadata['covariance'] = xcov
        state.metadata['lambda'] = lambda_
        
        # Update the proposal parameters
        self.mu = np.zeros((self.dim, 1))
        self.cov = lambda_ * xcov





        
        