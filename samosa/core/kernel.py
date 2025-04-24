"""
Template class file for the kernel
"""

# Imports
from typing import Protocol
from core.state import ChainState
from core.proposal import ProposalProtocol

class KernelProtocol(Protocol):
    """
    Protocol for MCMC transition kernels.
    """

    def propose(self, proposal: ProposalProtocol, state: 'ChainState') -> 'ChainState':
        """Generate candidate state from current state"""
        raise NotImplementedError("Implement propose method")
    
    def acceptance_ratio(self, proposal: ProposalProtocol, current: 'ChainState', proposed: 'ChainState') -> float:
        """Compute log acceptance probability"""
        raise NotImplementedError("Implement acceptance_ratio method")
    
    def adapt(self, history: list['ChainState']):
        """Update kernel parameters using chain history"""
        pass
