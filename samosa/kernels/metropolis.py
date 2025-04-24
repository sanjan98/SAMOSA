"""
Class file fpr the Metropolis-Hastings kernel
"""

# Imports
import numpy as np
from samosa.core.state import ChainState
from samosa.core.kernel import KernelProtocol
from samosa.core.proposal import ProposalProtocol
from samosa.core.model import ModelProtocol

class MetropolisHastingsKernel(KernelProtocol):
    """
    Metropolis-Hastings kernel for MCMC sampling.
    """

    def __init__(self, model: ModelProtocol):
        """
        Initialize the Metropolis-Hastings kernel with a model.
        """
        self.model = model

    def propose(self, proposal: ProposalProtocol, current_state: ChainState) -> ChainState:
        """
        Generate a candidate state from the current state using the proposal.
        """
        # Sample a new state using the proposal
        proposed_position = proposal.sample(current_state).position
        
        # Compute the attributes of the proposed state
        model_result = self.model(proposed_position)

        # Create a new ChainState object for the proposed state
        proposed_state = ChainState(position=proposed_position, **model_result, metadata=current_state.metadata.copy())
        
        return proposed_state
    
    def acceptance_ratio(self, proposal: ProposalProtocol, current: ChainState, proposed: ChainState) -> float:
        """
        Compute the log acceptance probability for the proposed state.
        """
        logq_forward, logq_reverse = proposal.proposal_logpdf(current, proposed)
        
        # Calculate the acceptance ratio
        check = (proposed.log_posterior + logq_reverse) - (current.log_posterior + logq_forward)
        if check > 0:
            ar = 1.0
        else:
            ar = np.exp(check)

        # Calculate the acceptance ratio
        return ar
    
    def adapt(self, proposal: ProposalProtocol, proposed: ChainState) -> None:
        """
        Adapt the proposal based on the proposed state.
        """

        # Check if the proposal has an adapt method
        if hasattr(proposal, 'adapt'):
            proposal.adapt(proposed)

        return None

    