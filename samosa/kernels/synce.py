"""
Class file for the (RE)SYNCE Coupled kernel for coupled chain
"""

# Imports
import numpy as np
from samosa.core.state import ChainState
from samosa.core.model import ModelProtocol
from samosa.core.proposal import ProposalProtocol
from samosa.core.kernel import KernelProtocol
from samosa.utils.tools import sample_multivariate_gaussian
from samosa.kernels.metropolis import MetropolisHastingsKernel
from samosa.proposals.gaussianproposal import IndependentProposal
from samosa.proposals.maximalproposal import MaximalCoupling
from typing import Tuple

class SYNCEKernel(KernelProtocol):
    """
    Coupled kernel for Multi-Level MCMC sampling.
    Maintains two chains (high-fidelity and low-fidelity) and proposes coupled moves.
    Also has the option of resyncjronization (RE)SYNCE. The resynchronization kernel is assumed to be the Independent Metropolis-Hastings kernel.
    The final RESYNCE kernel will be K_RESYNCE = (1-w)*K_SYNCE + w*K_RE, where K_SYNCE is the SYNCE kernel and K_RE is the resynchronization kernel.
    w is the weight for the resynchronization kernel, which can be set to 0 for the SYNCE kernel.
    """

    def __init__(self, coarse_model: ModelProtocol, fine_model: ModelProtocol, w: float = 0.0, resync_type: str = 'independent'):
        """
        Initialize the SYNCE Coupled kernel.
        
        Args:
            coarse_model: Low-fidelity model
            fine_model: High-fidelity model
            w: Weight for the resynchronization kernel (default is 0.0, which means no resynchronization)
            resync_type: Type of resynchronization kernel to use. Options are 'maximal', 'independent', or 'lot' (linear optimal transport).
        """
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.w = w
        self.resync_type = resync_type
        
    def propose(self, proposal_coarse: ProposalProtocol, proposal_fine: ProposalProtocol, current_coarse_state: ChainState, current_fine_state: ChainState) -> Tuple[ChainState, ChainState, ChainState, ChainState]:
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

        # Sample a uniform random number to decide whether to resynchronize or not
        u = np.random.rand()

        if u < self.w:

            # Resynchronization step
            # Depending on the type of resynchronization kernel, propose a common step
            if self.resync_type == 'maximal':
                maximal_proposal = MaximalCoupling(proposal_coarse, proposal_fine)
                proposed_fine_position, proposed_coarse_position = maximal_proposal.sample(current_coarse_state, current_fine_state)

            elif self.resync_type == 'independent':

                common_mean = (current_coarse_state.metadata['mean'] + current_fine_state.metadata['mean']) / 2
                common_cov = (current_coarse_state.metadata['covariance'] + current_fine_state.metadata['covariance']) / 2

                proposal = IndependentProposal(mu = common_mean, sigma = common_cov)

                eta = proposal.sample(current_fine_state)

                proposed_fine_position = eta.position
                proposed_coarse_position = eta.position

            elif self.resync_type == 'lot':

                # Use the linear optimal transport for two Gaussians
                eta = sample_multivariate_gaussian(np.zeros((dim, 1)), np.eye(dim))

                # Propose a move for the low-fidelity chain
                proposed_coarse_position = current_coarse_state.metadata['mean'] + np.linalg.cholesky(current_coarse_state.metadata['covariance']) @ eta
                # Propose a move for the high-fidelity chain
                proposed_fine_position = current_fine_state.metadata['mean'] + np.linalg.cholesky(current_fine_state.metadata['covariance']) @ eta

            else:
                raise ValueError(f"Unknown resynchronization type: {self.resync_type}. Supported types are 'maximal' and 'independent'.")
            
        else:
            # SYNCE step
            # Propose the common step from the standard Gaussian
            eta = sample_multivariate_gaussian(np.zeros((dim, 1)), np.eye(dim))

            # Propose a move for the low-fidelity chain
            proposed_coarse_position = proposal_coarse.sample(current_coarse_state, eta).position
            proposed_fine_position = proposal_fine.sample(current_fine_state, eta).position

        # Evaluate the low-fidelity model
        coarse_model_result = self.coarse_model(proposed_coarse_position)
        # Evaluate the high-fidelity model
        fine_model_result = self.fine_model(proposed_fine_position)

        # Create new ChainState objects for the proposed states
        proposed_coarse_state = ChainState(position=proposed_coarse_position, **coarse_model_result, metadata=current_coarse_state.metadata.copy())

        proposed_fine_state = ChainState(position=proposed_fine_position, **fine_model_result, metadata=current_fine_state.metadata.copy())
        
        return proposed_coarse_state, proposed_fine_state, current_coarse_state, current_fine_state
    
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