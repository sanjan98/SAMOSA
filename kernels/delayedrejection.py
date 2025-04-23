"""
Class file for the Delayed Rejection kernel
"""

# Imports
import numpy as np
from core.state import ChainState
from core.model import ModelProtocol
from core.proposal import ProposalProtocol
from core.kernel import KernelProtocol
from typing import Any, Dict, Callable, List, Optional

class DelayedRejectionKernel(KernelProtocol):
    """
    Delayed Rejection kernel for MCMC sampling.
    Implements multiple proposal stages, where each stage is tried 
    only if all previous stages have been rejected.
    """

    def __init__(self, model: ModelProtocol, cov_scale: float = 0.5):
        """
        Initialize the 2 stage Delayed Rejection kernel.
        
        Args:
            model: Function that computes log posterior given parameters
            cov_scale: Scaling factor for covariance of proposal distribution
        """
        self.model = model
        self.cov_scale = cov_scale
        # Keep track of all intermediate states for multi-stage acceptance
        self.first_stage_state: Optional[ChainState] = None
        
    def propose(self, proposal: ProposalProtocol, current_state: ChainState) -> ChainState:
        """
        Generate a candidate state from the current state using the proposal.
        This will either return the first stage proposal or the second stage proposal depending on the first stage acceptance ratio.
        """
        # Sample a new state using the proposal
        proposed_position = proposal.sample(current_state).position
        
        # Compute the attributes of the proposed state
        model_result = self.model(proposed_position)

        # Create a new ChainState object for the proposed state
        proposed_state = ChainState(position=proposed_position, **model_result, metadata=current_state.metadata.copy())
        
        # Store the first stage proposed state
        self.stage_states[0] = proposed_state
        
        return proposed_state
    
    def _generate_proposal(self, proposal: ProposalProtocol, current_state: ChainState) -> ChainState:
        """Generate the first stage proposal"""
        proposed_position = proposal.sample(current_state).position
        model_result = self.model(proposed_position)

        return ChainState(position=proposed_position, **model_result, metadata=current_state.metadata.copy())
    
    
        """
        Complete delayed rejection sampling logic - this would be used by the sampler
        to enable the multi-stage proposal process.
        """
        # First stage proposal
        proposed_state = self.propose(proposal, current_state)
        ar = self.acceptance_ratio(proposal, current_state, proposed_state)
        
        # Accept/reject first stage
        if np.random.rand() < ar:
            return proposed_state
            
        # If rejected, try additional stages
        for stage in range(1, self.max_stages):
            # Generate another proposal
            proposed_state = self.propose_stage(proposal, current_state, stage)
            self.stage_states[stage] = proposed_state
            
            # Calculate delayed rejection acceptance ratio
            ar = self.acceptance_ratio(proposal, current_state, proposed_state)
            
            # Accept/reject this stage
            if np.random.rand() < ar:
                return proposed_state
                
        # If all proposals rejected, stay at current state
        return current_state