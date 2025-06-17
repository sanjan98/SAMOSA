"""
Class file for the Transport SYNCE Coupled kernel for coupled chain
"""

# Imports
import numpy as np
from samosa.core.state import ChainState
from samosa.core.model import ModelProtocol
from samosa.core.proposal import ProposalProtocol
from samosa.core.kernel import KernelProtocol
from samosa.utils.tools import sample_multivariate_gaussian
from samosa.kernels.metropolis import MetropolisHastingsKernel
from typing import Tuple, Dict, List, Any

class TransportSYNCEKernel(KernelProtocol):
    """
    Coupled kernel using Transport maps for Multi-Level MCMC sampling.
    Maintains two chains (high-fidelity and low-fidelity) and proposes coupled moves.
    """

    def __init__(self, coarse_model: ModelProtocol, fine_model: ModelProtocol, coarse_map: Any, fine_map: Any, w: float = 0.0, coupletype: str = 'direct'):
        """
        Initialize the SYNCE Coupled kernel.
        
        Args:
            coarse_model: Low-fidelity model
            fine_model: High-fidelity model
            coarse_map: Transport map for the low-fidelity model
            fine_map: Transport map for the high-fidelity model
            w: Weight for the resynchronization kernel (default is 0.0, which means no resynchronization)
        """
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.coarse_map = coarse_map
        self.fine_map = fine_map
        self.w = w
        if coupletype not in ['deep', 'direct']:
            raise ValueError("coupletype must be either 'deep' or 'direct'")
        self.coupletype = coupletype
        
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

        coarse_theta = current_coarse_state.position
        fine_theta = current_fine_state.position

        coarse_r, logdet_current_coarse = self.coarse_map.forward(coarse_theta)

        # Bring the corase sample to the reference space
        # Depending on type of couplingtype, bring it back to the right space
        if self.coupletype == 'deep':
            
            # Bring the fine sample to the coarse space
            ftoc_theta, logdet_ftoc_fine = self.fine_map.forward(fine_theta)
            fine_r, logdet_ctor_fine = self.coarse_map.forward(ftoc_theta)
            logdet_current_fine = logdet_ftoc_fine + logdet_ctor_fine

        elif self.coupletype == 'direct':

            # Bring the fine sample to the reference space
            fine_r, logdet_current_fine = self.fine_map.forward(fine_theta)

        # Propose the common step from the standard Gaussian
        eta = sample_multivariate_gaussian(np.zeros((dim, 1)), np.eye(dim))

        # Sample a uniform random number to decide whether to resynchronize or not
        u = np.random.rand()

        if u < self.w:
            # Resynchronization step
            # Propose a move for the low-fidelity chain
            coarse_rprime = np.linalg.cholesky(proposal_coarse.cov) @ eta if hasattr(proposal_coarse, 'cov') else np.linalg.cholesky(proposal_coarse.proposal.cov) @ eta
            # Propose a move for the high-fidelity chain
            fine_rprime = np.linalg.cholesky(proposal_fine.cov) @ eta if hasattr(proposal_fine, 'cov') else np.linalg.cholesky(proposal_fine.proposal.cov) @ eta

        else:
            # If no resynchronization, propose a move for both chains independently
            # Propose a move for the low-fidelity chain
            coarse_rprime = coarse_r + np.linalg.cholesky(proposal_coarse.cov) @ eta if hasattr(proposal_coarse, 'cov') else coarse_r + np.linalg.cholesky(proposal_coarse.proposal.cov) @ eta
            # Propose a move for the high-fidelity chain
            fine_rprime = fine_r + np.linalg.cholesky(proposal_fine.cov) @ eta if hasattr(proposal_fine, 'cov') else fine_r + np.linalg.cholesky(proposal_fine.proposal.cov) @ eta

        # Bring the coarse sample back to the original space
        coarse_thetaprime, logdet_proposed_coarse_temp = self.coarse_map.inverse(coarse_rprime)
        logdet_proposed_coarse = -logdet_proposed_coarse_temp

        # Bring the fine sample back to the original space
        if self.coupletype == 'deep':
            rtoc_tetaprime, logdet_rtoc_fine = self.coarse_map.inverse(fine_rprime)
            fine_thetaprime, logdet_ctof_fine = self.fine_map.inverse(rtoc_tetaprime)
            logdet_proposed_fine = -(logdet_rtoc_fine + logdet_ctof_fine)

        elif self.coupletype == 'direct':
            fine_thetaprime, logdet_proposed_fine_temp = self.fine_map.inverse(fine_rprime)
            logdet_proposed_fine = -logdet_proposed_fine_temp

        # Evaluate the low-fidelity model
        coarse_model_result = self.coarse_model(coarse_thetaprime)
        # Evaluate the high-fidelity model
        fine_model_result = self.fine_model(fine_thetaprime)

        # Create new ChainState objects for the proposed states
        proposed_coarse_state = ChainState(position=coarse_thetaprime, reference_position=coarse_rprime, **coarse_model_result, metadata=current_coarse_state.metadata.copy())

        proposed_fine_state = ChainState(position=fine_thetaprime, reference_position=fine_rprime, **fine_model_result, metadata=current_fine_state.metadata.copy())

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
        
        coarse_r, logdet_current_coarse = self.coarse_map.forward(current_coarse.position)
        coarse_rprime, logdet_proposed_coarse = self.coarse_map.forward(proposed_coarse.position)

        current_reference_coarse = ChainState(position=coarse_r, log_posterior=None)
        proposed_reference_coarse = ChainState(position=coarse_rprime, log_posterior=None)

        logq_forward_coarse, logq_reverse_coarse = proposal_coarse.proposal_logpdf(current_reference_coarse, proposed_reference_coarse)
        
        # Calculate the acceptance ratio
        check_coarse = (proposed_coarse.log_posterior + logq_reverse_coarse - logdet_proposed_coarse) - (current_coarse.log_posterior + logq_forward_coarse - logdet_current_coarse)
        if check_coarse > 0:
            ar_coarse = 1.0
        else:
            ar_coarse = np.exp(check_coarse)

        # Implementing only direct mapping for now
        fine_r, logdet_current_fine = self.fine_map.forward(current_fine.position)
        fine_rprime, logdet_proposed_fine = self.fine_map.forward(proposed_fine.position)
        current_reference_fine = ChainState(position=fine_r, log_posterior=None)
        proposed_reference_fine = ChainState(position=fine_rprime, log_posterior=None)

        logq_forward_fine, logq_reverse_fine = proposal_fine.proposal_logpdf(current_reference_fine, proposed_reference_fine)

        # Calculate the acceptance ratio
        check_fine = (proposed_fine.log_posterior + logq_reverse_fine - logdet_proposed_fine) - (current_fine.log_posterior + logq_forward_fine - logdet_current_fine)
        if check_fine > 0:
            ar_fine = 1.0
        else:
            ar_fine = np.exp(check_fine)

        # Calculate the acceptance ratio
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
    
    def adapt_maps(self, samples_coarse: List[ChainState], samples_fine: List[ChainState]) -> None:
        """
        Adapt the transport maps based on the samples.

        Parameters:
            samples_coarse: List of samples from the low-fidelity chain
            samples_fine: List of samples from the high-fidelity chain

        Returns:
            None
        """
        
        self.coarse_map.adapt(samples_coarse)
        self.fine_map.adapt(samples_fine)
        return None