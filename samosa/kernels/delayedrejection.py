"""
Class file for the Delayed Rejection kernel
"""

# Imports
import numpy as np
from samosa.core.state import ChainState
from samosa.core.model import ModelProtocol
from samosa.core.proposal import ProposalProtocol
from samosa.core.kernel import KernelProtocol
from typing import Optional

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

    def propose(self, proposal: ProposalProtocol, current: 'ChainState') -> 'ChainState':
        
        proposed_state1 = self._proposestate(proposal, current)
        # Store the first stage state
        self.first_stage_state = proposed_state1
        # Check if the first stage is accepted
        ar1 = self.acceptance_ratio(proposal, current, proposed_state1)

        u = np.random.rand()
        if ar1 == 1.0 or u < ar1:
            # Accept the first stage
            self.ar = ar1 # Store the acceptance ratio
            return proposed_state1
        else:
            # If the first stage is rejected, propose a second stage
            # Temporarily scale the covariance
            if hasattr(proposal, 'cov'):
                original_cov = proposal.cov.copy()
                proposal.cov = original_cov * self.cov_scale
                proposed_state2 = self._proposestate(proposal, current)
                proposal.cov = original_cov

            elif hasattr(proposal.proposal, 'cov'):
                original_cov = proposal.proposal.cov.copy()
                proposal.proposal.cov = original_cov * self.cov_scale
                proposed_state2 = self._proposestate(proposal, current)
                proposal.proposal.cov = original_cov

            else:
                proposed_state2 = self._proposestate(proposal, current)

            # Compute the acceptance ratio for the second stage
            ar2 = self._second_stage_acceptance_ratio(proposal, current, proposed_state1, proposed_state2)

            self.ar = ar2 # Store the acceptance ratio
            
            # Accept or reject the second stage
            u2 = np.random.rand()
            if ar2 == 1.0 or u2 < ar2:
                # Accept the second stage
                return proposed_state2
            else:
                # Reject both stages
                return current
    
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

    def _second_stage_acceptance_ratio(self, proposal: ProposalProtocol, current: ChainState, first_stage: ChainState, second_stage: ChainState) -> float:
        """
        Calculate the delayed rejection acceptance ratio for the second stage.
        
        Args:
            proposal: The proposal distribution
            current: The current chain state
            first_stage: The first stage proposed state (rejected)
            second_stage: The second stage proposed state
            
        Returns:
            The acceptance ratio for the second stage
        """
        # Calculate standard proposal densities
        logq_forward_1, logq_reverse_1 = proposal.proposal_logpdf(current, first_stage)
        
        # For the second stage, we need to scale the proposal temporarily
        if hasattr(proposal, 'cov'):
            original_cov = proposal.cov.copy()
            proposal.cov = original_cov * self.cov_scale
            
            # Calculate second-order proposal densities
            logq_forward_2, logq_reverse_2 = proposal.proposal_logpdf(current, second_stage)
            
            # Calculate hypothetical proposal densities from second_stage to first_stage
            logq_y2_to_y1, logq_y1_to_y2 = proposal.proposal_logpdf(second_stage, first_stage)
            
            # Restore the original covariance
            proposal.cov = original_cov

        elif hasattr(proposal.proposal, 'cov'):
            original_cov = proposal.proposal.cov.copy()
            proposal.proposal.cov = original_cov * self.cov_scale
            
            # Calculate second-order proposal densities
            logq_forward_2, logq_reverse_2 = proposal.proposal_logpdf(current, second_stage)
            
            # Calculate hypothetical proposal densities from second_stage to first_stage
            logq_y2_to_y1, logq_y1_to_y2 = proposal.proposal_logpdf(second_stage, first_stage)
            
            # Restore the original covariance
            proposal.proposal.cov = original_cov

        else:
            # If proposal doesn't have adjustable covariance
            logq_forward_2, logq_reverse_2 = proposal.proposal_logpdf(current, second_stage)
            logq_y2_to_y1, logq_y1_to_y2 = proposal.proposal_logpdf(second_stage, first_stage)
        
        # Calculate first stage rejection probability
        check_alpha_1 = (first_stage.log_posterior - current.log_posterior) + (logq_reverse_1 - logq_forward_1)
        if check_alpha_1 > 0:
            alpha_1 = 1.0
        else:
            alpha_1 = np.exp(check_alpha_1)

        # Calculate hypothetical reverse first stage rejection probability
        check_alpha_1_reverse = (first_stage.log_posterior - second_stage.log_posterior) + (logq_y1_to_y2 - logq_y2_to_y1)
        if check_alpha_1_reverse > 0:
            alpha_1_reverse = 1.0
        else:
            alpha_1_reverse = np.exp(check_alpha_1_reverse)    

        # Calculate the numerator and denominator terms
        numerator = (1 - alpha_1) * np.exp((second_stage.log_posterior - current.log_posterior) + (logq_reverse_2 - logq_forward_2))
        denominator = 1 - alpha_1_reverse
        
        # Avoid division by zero
        if denominator < 1e-10:
            return 0.0
            
        # Compute the final acceptance ratio
        ar = min(1.0, numerator / denominator)
        return ar    
    
    def adapt(self, proposal: ProposalProtocol, proposed: ChainState) -> None:
        """
        Adapt the proposal based on the proposed state.
        """

        # Check if the proposal has an adapt method
        if hasattr(proposal, 'adapt'):
            proposal.adapt(proposed)

        return None
    
    def _proposestate(self, proposal: ProposalProtocol, current_state: ChainState) -> ChainState:
        
        # Sample a new state using the proposal
        proposed_position = proposal.sample(current_state).position
        
        # Compute the attributes of the proposed state
        model_result = self.model(proposed_position)

        # Create a new ChainState object for the proposed state
        proposed_state = ChainState(position=proposed_position, **model_result, metadata=current_state.metadata.copy())
        
        return proposed_state