"""
Template class file for the kernel
"""

# Imports
from abc import ABC, abstractmethod
from core.state import ChainState
from core.proposal import Proposal

class TransitionKernel(ABC):
    """
    Base class for MCMC transition kernels.
    """

    @abstractmethod
    def propose(self, proposal: Proposal, state: 'ChainState') -> 'ChainState':
        """Generate candidate state from current state"""
        raise NotImplementedError("Implement propose method")
    
    @abstractmethod
    def acceptance_ratio(self, proposal: Proposal, current: 'ChainState', proposed: 'ChainState') -> float:
        """Compute log acceptance probability"""
        raise NotImplementedError("Implement acceptance_ratio method")
    
    def adapt(self, history: list['ChainState']):
        """Update kernel parameters using chain history"""
        pass
