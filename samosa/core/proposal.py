"""
Template class file for proposal
"""

# Imports
import numpy as np
from typing import Protocol, Optional, Tuple
from samosa.core.state import ChainState

class ProposalProtocol(Protocol):
    """
    Protocol for proposal distributions
    """

    def sample(self, current_state: 'ChainState', common_step: Optional[np.ndarray] = None) -> 'ChainState':
        """Generate candidate state from current state"""
        raise NotImplementedError("Implement sample method")
    
    def proposal_logpdf(self, current_state: 'ChainState', proposed_state: 'ChainState') -> float:
        """Compute forward (proposed state given current state) and reverse (current state given proposed state) log probability"""
        raise NotImplementedError("Implement proposal_logpdf method")
    
    def adapt(self, state: 'ChainState') -> None:
        """Adapt the proposal distribution based on the current state"""
        pass

# Adapter Protocol
class AdapterBase:
    """Base class for proposal adaptation strategies"""

    def __init__(self, adapt_start: int = 500, adapt_end: int = 1000, eps: float = 1e-06):
        self.adapt_start = adapt_start
        self.adapt_end = adapt_end
        self.eps = eps
    
    def adapt(self, proposal: ProposalProtocol, state: ChainState) -> None:
        """Adapt the proposal based on the current state"""
        raise NotImplementedError("Subclass must implement adapt method")

# Adaptive Proposal
# Proposal wrapper that can optionally use adaptation
class AdaptiveProposal(ProposalProtocol):
    """Proposal wrapper that can optionally use adaptation"""
    
    def __init__(self, base_proposal: ProposalProtocol, adapter: Optional[AdapterBase] = None):
        self.proposal = base_proposal
        self.adapter = adapter
        
    def sample(self, current_state: ChainState) -> ChainState:
        """Sample from the base proposal"""
        return self.proposal.sample(current_state)
    
    def proposal_logpdf(self, current_state: ChainState, proposed_state: ChainState) -> Tuple[float, float]:
        """Calculate forward and reverse log probability using the base proposal"""
        return self.proposal.proposal_logpdf(current_state, proposed_state)
    
    def adapt(self, state: ChainState) -> None:
        """Adapt the proposal if an adapter is provided"""
        if self.adapter is not None:
            self.adapter.adapt(self.proposal, state)