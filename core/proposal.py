"""
Template class file for proposal
"""

# Imports
from abc import ABC, abstractmethod
from core.state import ChainState

class Proposal(ABC):
    """
    Base class for the proposal
    """

    @abstractmethod
    def sample(self, current_state: 'ChainState') -> 'ChainState':
        """Generate candidate state from current state"""
        raise NotImplementedError("Implement sample method")
    
    @abstractmethod
    def proposal_logpdf(self, current_state: 'ChainState', proposed_state: 'ChainState') -> float:
        """Compute forward (proposed state given current state) and reverse (current state given proposed state) log probability"""
        raise NotImplementedError("Implement proposal_logpdf method")