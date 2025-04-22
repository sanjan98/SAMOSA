"""
Class file fpr the Metropolis-Hastings kernel
"""

# Imports
import numpy as np
from core.state import ChainState
from core.kernel import TransitionKernel
from core.proposal import Proposal
from core.model import BaseModel

class MetropolisHastingsKernel(TransitionKernel):
    """
    Metropolis-Hastings kernel for MCMC sampling.
    """

    def __init__(self, model: BaseModel):
        """
        Initialize the Metropolis-Hastings kernel with a model.
        """
        self.model = model

    def propose(self, proposal: Proposal, current_state: ChainState) -> ChainState:
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
    
    def acceptance_ratio(self, proposal: Proposal, current: ChainState, proposed: ChainState) -> float:
        """
        Compute the log acceptance probability for the proposed state.
        """
        logq_forward, logq_reverse = proposal.proposal_logpdf(current, proposed)
        
        # Calculate the acceptance ratio
        return (proposed.log_posterior + logq_reverse) - (current.log_posterior + logq_forward)
    
    # STOPPED HERE!!
    # I am confused if the updates should happen here because the kernel is also being adapted as it uses the proposal. Maybe check if proposal class has an update method and call it here if it does. Then the sampler class always calls the adapt method of the kernel and the kernel calls the adapt method of the proposal if it exists. But how do we pass the information required for the different udpates from the sampler to the kernel?