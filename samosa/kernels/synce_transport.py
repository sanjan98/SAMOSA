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

    def __init__(self, coarse_model: ModelProtocol, fine_model: ModelProtocol, coarse_map: Any, fine_map: Any, coupletype: str = 'deep'):
        """
        Initialize the SYNCE Coupled kernel.
        
        Args:
            coarse_model: Low-fidelity model
            fine_model: High-fidelity model
        """
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.coarse_map = coarse_map
        self.fine_map = fine_map
        if coupletype not in ['deep', 'direct']:
            raise ValueError("coupletype must be either 'deep' or 'shallow'")
        self.coupletype = coupletype
        
    def propose(self, proposal_coarse: ProposalProtocol, proposal_fine: ProposalProtocol, current_coarse_state: ChainState, current_fine_state: ChainState) -> Tuple[ChainState, ChainState, float, float, float, float]:
        """
        Generate a candidate state for both chains.
        
        Parameters:
            proposal_coarse: Proposal distribution for the low-fidelity chain
            proposal_fine: Proposal distribution for the high-fidelity chain
            current_coarse_state: Current state of the low-fidelity chain
            current_fine_state: Current state of the high-fidelity chain
            
        Returns:
            Tuple of proposed states for the low-fidelity and high-fidelity chains, along with log determinants
        """

        dim = current_coarse_state.position.shape[0]
        assert dim == current_fine_state.position.shape[0], "The dimensions of the two chains must be the same."

        coarse_theta = current_coarse_state.position
        fine_theta = current_fine_state.position

        coarse_r, logdet_current_coarse = self.coarse_map.forward(coarse_theta.position)

        # Bring the corase sample to the reference space
        # Depending on type of couplingtype, bring it back to the right space
        if self.coupletype == 'deep':
            
            # Bring the fine sample to the coarse space
            ftoc_theta, logdet_current_ftoc = self.fine_map.forward(fine_theta.position)
            fine_r, logdet_current_ctor = self.coarse_map.forward(ftoc_theta)
            logdet_current_fine = logdet_current_ftoc + logdet_current_ctor

        elif self.coupletype == 'direct':

            # Bring the fine sample to the reference space
            fine_r, logdet_current_fine = self.fine_map.forward(fine_theta.position)

        # Propose the common step from the standard Gaussian
        eta = sample_multivariate_gaussian(np.zeros((dim, 1)), np.eye(dim))

        # Propose a move for the low-fidelity chain
        coarse_rprime = coarse_r + np.linalg.cholesky(proposal_coarse.cov) @ eta if hasattr(proposal_coarse, 'cov') else coarse_r + np.linalg.cholesky(proposal_coarse.proposal.cov) @ eta
        # Propose a move for the high-fidelity chain
        fine_rprime = fine_r + np.linalg.cholesky(proposal_fine.cov) @ eta if hasattr(proposal_fine, 'cov') else fine_r + np.linalg.cholesky(proposal_fine.proposal.cov) @ eta

        # Bring the coarse sample back to the original space
        coarse_thetaprime, logdet_proposed_coarse = self.coarse_map.inverse(coarse_rprime)

        # Bring the fine sample back to the original space
        if self.coupletype == 'deep':
            rtoc_tetaprime, logdet_proposed_rtoc = self.coarse_map.inverse(fine_rprime)
            fine_thetaprime, logdet_proposed_ctof = self.fine_map.inverse(rtoc_tetaprime)
            logdet_proposed_fine = logdet_proposed_rtoc + logdet_proposed_ctof

        elif self.coupletype == 'direct':
            fine_thetaprime, logdet_proposed_fine = self.fine_map.inverse(fine_rprime)

        # Evaluate the low-fidelity model
        coarse_model_result = self.coarse_model(coarse_thetaprime)
        # Evaluate the high-fidelity model
        fine_model_result = self.fine_model(fine_thetaprime)

        # Create new ChainState objects for the proposed states
        proposed_coarse_state = ChainState(position=coarse_thetaprime, **coarse_model_result, metadata=current_coarse_state.metadata.copy())

        proposed_fine_state = ChainState(position=fine_thetaprime, **fine_model_result, metadata=current_fine_state.metadata.copy())

        # This might be wrong, please check
        # Add the log determinants to the correct logposteriors
        proposed_coarse_state.log_posterior += logdet_proposed_coarse
        proposed_fine_state.log_posterior += logdet_proposed_fine

        # Update the current logposteriors as well
        # This is needed for the acceptance ratio
        current_coarse_state.log_posterior += -logdet_current_coarse         
        current_fine_state.log_posterior += -logdet_current_fine

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