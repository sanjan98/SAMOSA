"""
Base classes for MCMC proposal distributions and adaptation strategies.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Any
from samosa.core.state import ChainState


class ProposalBase(ABC):
    """
    Abstract base class for MCMC proposal distributions.
    
    All proposals must implement sample() and proposal_logpdf() methods.
    Proposals can optionally implement adapt() for self-adaptation.
    
    Attributes:
        mu (np.ndarray): Mean vector of the proposal distribution.
        cov (np.ndarray): Covariance matrix of the proposal distribution.
    """

    def __init__(self, mu: np.ndarray, cov: np.ndarray) -> None:
        """
        Initialize proposal with mean and covariance.
        
        Args:
            mu: Mean vector (d, 1).
            cov: Covariance matrix (d, d).
        """
        self.mu = mu
        self.cov = cov

    @abstractmethod
    def sample(self, current_state: ChainState, common_step: Optional[np.ndarray] = None) -> ChainState:
        """
        Generate a candidate state from the proposal distribution.
        
        Args:
            current_state: Current state of the chain.
            common_step: Optional common random variable for coupling.
            
        Returns:
            Proposed ChainState.
        """
        raise NotImplementedError("Subclasses must implement sample()")
    
    @abstractmethod
    def proposal_logpdf(self, current_state: ChainState, proposed_state: ChainState) -> tuple[float, float]:
        """
        Compute forward and reverse log probabilities.
        
        Args:
            current_state: Current state of the chain.
            proposed_state: Proposed state.
            
        Returns:
            Tuple of (log_q_forward, log_q_reverse) where:
                - log_q_forward: log q(proposed | current)
                - log_q_reverse: log q(current | proposed)
        """
        raise NotImplementedError("Subclasses must implement proposal_logpdf()")
    
    def update_parameters(self, mu: Optional[np.ndarray] = None, cov: Optional[np.ndarray] = None) -> None:
        """
        Update proposal parameters (called by adapters).
        
        Args:
            mu: New mean vector (optional).
            cov: New covariance matrix (optional).
        """
        if mu is not None:
            self.mu = mu
        if cov is not None:
            self.cov = cov

    def adapt(self, state: ChainState) -> None:
        """
        Self-adapt proposal based on current state (optional).
        
        Override this method if the proposal has its own adaptation logic
        (e.g., adaptive beta in pCN proposals).
        
        Args:
            state: Current state containing metadata for adaptation.
        """
        pass


class AdapterBase(ABC):
    """
    Abstract base class for proposal adaptation strategies.
    
    Adapters modify proposal parameters based on chain history to
    improve sampling efficiency (e.g., Haario covariance adaptation).
    
    Attributes:
        adapt_start: Iteration to start adaptation.
        adapt_end: Iteration to end adaptation.
        eps: Regularization parameter for covariance updates.
    """

    def __init__(self, adapt_start: int = 500, adapt_end: int = 1000, eps: float = 1e-06) -> None:
        """
        Initialize adapter with adaptation window and regularization.
        
        Args:
            adapt_start: Start adaptation at this iteration.
            adapt_end: Stop adaptation after this iteration.
            eps: Small constant for numerical stability.
        """
        if adapt_start < 0 or adapt_end < adapt_start:
            raise ValueError("Invalid adaptation window: must have 0 <= adapt_start <= adapt_end")
        
        self.adapt_start = adapt_start
        self.adapt_end = adapt_end
        self.eps = eps
    
    @abstractmethod
    def adapt(self, proposal: ProposalBase, state: ChainState) -> None:
        """
        Adapt proposal parameters based on current state.
        
        Args:
            proposal: The proposal distribution to adapt.
            state: Current state containing adaptation metadata.
        """
        raise NotImplementedError("Subclasses must implement adapt()")


class AdaptiveProposal:
    """
    Wrapper that applies an adapter to a base proposal.
    
    Delegates all proposal operations to the wrapped base proposal,
    but uses the adapter's strategy for parameter updates.
    
    Attributes:
        proposal: The wrapped base proposal.
        adapter: The adaptation strategy.
    """
    
    def __init__(self, base_proposal: ProposalBase, adapter: AdapterBase) -> None:
        """
        Wrap a proposal with an adapter.
        
        Args:
            base_proposal: The proposal to wrap.
            adapter: The adaptation strategy to use.
        """
        self.proposal = base_proposal
        self.adapter = adapter

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped proposal."""
        return getattr(self.proposal, name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate attribute setting to wrapped proposal, except wrapper attributes."""
        if name in ('proposal', 'adapter'):
            super().__setattr__(name, value)
        else:
            setattr(self.proposal, name, value)
    
    def adapt(self, state: ChainState) -> None:
        """Use the adapter to adapt the wrapped proposal."""
        self.adapter.adapt(self.proposal, state)
        self.proposal.adapt(state)