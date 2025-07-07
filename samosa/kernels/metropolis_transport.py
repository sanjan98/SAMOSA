"""
Class file for the Metropolis-Hastings kernel with transport maps
"""

# Imports
import numpy as np
from samosa.core.state import ChainState
from samosa.core.kernel import KernelProtocol
from samosa.core.proposal import ProposalProtocol
from samosa.core.model import ModelProtocol

from dataclasses import replace
from typing import Any, List

class TransportMetropolisHastingsKernel(KernelProtocol):
    """
    Metropolis-Hastings kernel for MCMC sampling with transport maps.
    """

    def __init__(self, model: ModelProtocol, map: Any):
        """
        Initialize the Metropolis-Hastings kernel with a model.
        """
        self.model = model
        self.map = map

    def propose(self, proposal: ProposalProtocol, current_state: ChainState) -> ChainState:
        """
        Generate a candidate state from the current state using the map with a proposal.

        Parameters:
            proposal: Proposal distribution
            current_state: Current state of the chain
        Returns:
            proposed_state: Proposed state
        """

        # Get the current position in the reference space
        r, logdet_current = self.map.forward(current_state.position)
        # Update the current state with the reference position
        current_state = replace(current_state, reference_position=r)
        current_state.metadata['logdetT'] = logdet_current

        # Dummy state for the proposal
        current_reference = ChainState(position=r, log_posterior=None)
       
        # Sample a new state using the proposal
        rprime = proposal.sample(current_reference).position

        # Send the proposed state back to the original space
        proposed_position, logdet_proposed = self.map.inverse(rprime)
        
        # Compute the attributes of the proposed state
        model_result = self.model(proposed_position)

        # Create a new ChainState object for the proposed state
        proposed_state = ChainState(position=proposed_position, reference_position=rprime, **model_result, metadata=current_state.metadata.copy())
        proposed_state.metadata['logdetT'] = -logdet_proposed
        
        return proposed_state, current_state
    
    def acceptance_ratio(self, proposal: ProposalProtocol, current: ChainState, proposed: ChainState) -> float:
        """
        Compute the log acceptance probability for the proposed state.

        Parameters:
            proposal: Proposal distribution
            current: Current state of the chain
            proposed: Proposed state of the chain
        Returns:
            ar: Acceptance ratio
        """

        r = current.reference_position
        rprime = proposed.reference_position
        logdet_current = current.metadata['logdetT']
        logdet_proposed = proposed.metadata['logdetT']

        current_reference = ChainState(position=r, log_posterior=None)
        proposed_reference = ChainState(position=rprime, log_posterior=None)

        logq_forward, logq_reverse = proposal.proposal_logpdf(current_reference, proposed_reference)
        
        # Calculate the acceptance ratio
        check = (proposed.log_posterior + logq_reverse - logdet_proposed) - (current.log_posterior + logq_forward - logdet_current)
        if check > 0:
            ar = 1.0
        else:
            ar = np.exp(check)

        # Calculate the acceptance ratio
        return ar
    
    def adapt(self, proposal: ProposalProtocol, proposed: ChainState) -> None:
        """
        Adapt the proposal based on the proposed state.

        Parameters:
            proposal: Proposal distribution
            proposed: Proposed state of the chain
        Returns:
            None
        """

        # Check if the proposal has an adapt method
        if hasattr(proposal, 'adapt'):
            proposal.adapt(proposed)

        return None
    
    def adapt_map(self, samples: List[ChainState]) -> None:
        """
        Adapt the transport map based on the samples.

        Parameters:
            samples: List of ChainState objects
        Returns:
            None
        """
        if hasattr(self.map, 'adapt'):
            self.map.adapt(samples)
        
        return None

    def save_map(self, output_dir: str, iteration: int) -> None:
        """
        Save the transport map to a file.

        Parameters:
            output_dir: Directory to save the map
            iteration: Current iteration number
        Returns:
            None
        """
        if hasattr(self.map, 'checkpoint_model'):
            self.map.checkpoint_model(f'{output_dir}/map_{iteration}')
        
        return None

    