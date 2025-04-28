"""
Class file for the SYNCE Coupled kernel for coupled chain
"""

# Imports
import numpy as np
from samosa.core.state import ChainState
from samosa.core.model import ModelProtocol
from samosa.core.proposal import ProposalProtocol
from samosa.core.kernel import KernelProtocol
from samosa.utils.tools import sample_multivariate_gaussian
from samosa.kernels.metropolis import MetropolisHastingsKernel
from typing import Tuple

class SYNCEKernel(KernelProtocol):
    """
    Coupled kernel for Multi-Level MCMC sampling.
    Maintains two chains (high-fidelity and low-fidelity) and proposes coupled moves.
    """

    def __init__(self, coarse_model: ModelProtocol, fine_model: ModelProtocol):
        """
        Initialize the SYNCE Coupled kernel.
        
        Args:
            coarse_model: Low-fidelity model
            fine_model: High-fidelity model
        """
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        
    def propose(self, proposal_coarse: ProposalProtocol, proposal_fine: ProposalProtocol, current_coarse_state: ChainState, current_fine_state: ChainState) -> Tuple[ChainState, ChainState]:
        """
        Generate a candidate state for both chains.
        
        Parameters:
            proposal_coarse: Proposal distribution for the low-fidelity chain
            proposal_fine: Proposal distribution for the high-fidelity chain
            current_coarse_state: Current state of the low-fidelity chain
            current_fine_state: Current state of the high-fidelity chain
            
        Returns:
            Tuple of proposed states for the low-fidelity and high-fidelity chains
        """

        dim = current_coarse_state.position.shape[0]
        assert dim == current_fine_state.position.shape[0], "The dimensions of the two chains must be the same."
        
        # Propose the common step from the standard Gaussian
        eta = sample_multivariate_gaussian(np.zeros((dim, 1)), np.eye(dim))

        # Propose a move for the low-fidelity chain
        proposed_coarse_position = current_coarse_state.position + np.linalg.cholesky(proposal_coarse.cov) @ eta if hasattr(proposal_coarse, 'cov') else current_coarse_state.position + np.linalg.cholesky(proposal_coarse.proposal.cov) @ eta
        # Propose a move for the high-fidelity chain
        proposed_fine_position = current_fine_state.position + np.linalg.cholesky(proposal_fine.cov) @ eta if hasattr(proposal_fine, 'cov') else current_fine_state.position + np.linalg.cholesky(proposal_fine.proposal.cov) @ eta

        # Evaluate the low-fidelity model
        coarse_model_result = self.coarse_model(proposed_coarse_position)
        # Evaluate the high-fidelity model
        fine_model_result = self.fine_model(proposed_fine_position)

        # Create new ChainState objects for the proposed states
        proposed_coarse_state = ChainState(position=proposed_coarse_position, **coarse_model_result, metadata=current_coarse_state.metadata.copy())

        proposed_fine_state = ChainState(position=proposed_fine_position, **fine_model_result, metadata=current_fine_state.metadata.copy())
        
        return proposed_coarse_state, proposed_fine_state
    
    def acceptance_ratio(self, proposal_coarse: ProposalProtocol, current_coarse: ChainState, proposed_coarse: ChainState, proposal_fine: ProposalProtocol, current_fine: ChainState, proposed_fine: ChainState) -> Tuple[float, float]:
        """
        Return the acceptance ratio for both the chains.

        Parameters:
            proposal_coarse: Proposal distribution for the low-fidelity chain
            current_coarse: Current state of the low-fidelity chain
            proposed_coarse: Proposed state of the low-fidelity chain
            proposal_fine: Proposal distribution for the high-fidelity chain
            current_fine: Current state of the high-fidelity chain
            proposed_fine: Proposed state of the high-fidelity chain

        Returns:
            Tuple of acceptance ratios for the low-fidelity and high-fidelity chains
        """
        
        ar_coarse = MetropolisHastingsKernel(self.coarse_model).acceptance_ratio(proposal_coarse, current_coarse, proposed_coarse)

        ar_fine = MetropolisHastingsKernel(self.fine_model).acceptance_ratio(proposal_fine, current_fine, proposed_fine)

        return ar_coarse, ar_fine
    
    def adapt(self, proposal_coarse: ProposalProtocol, proposed_coarse: ChainState, proposal_fine: ProposalProtocol, proposed_fine: ChainState) -> None:
        """
        Adapt both the proposals based on their state.

        Parameters:
            proposal_coarse: Proposal distribution for the low-fidelity chain
            proposed_coarse: Proposed state of the low-fidelity chain
            proposal_fine: Proposal distribution for the high-fidelity chain
            proposed_fine: Proposed state of the high-fidelity chain

        Returns:
            None
        """
        # Check if the proposals have an adapt method
        if hasattr(proposal_coarse, 'adapt'):
            proposal_coarse.adapt(proposed_coarse)

        if hasattr(proposal_fine, 'adapt'):
            proposal_fine.adapt(proposed_fine)
        
        return None