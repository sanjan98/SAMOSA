"""
Class file for a single chain MCMC sampler.
"""

from samosa.core.model import ModelProtocol
from samosa.core.state import ChainState
from samosa.core.kernel import KernelProtocol
from samosa.core.proposal import ProposalProtocol
from typing import List, Tuple
import numpy as np
import os
import copy

import pickle

class coupledMCMCsampler:
    """
    Class for a coupled chain MCMC sampler.
    
    Attributes:
        kernel (TransitionKernel): The transition kernel used for sampling.
        proposal (Proposal): The proposal distribution used for generating candidate states.
        model (BaseModel): The model used for computing posterior values.
        initial_state (ChainState): The initial state of the chain.
        n_iterations (int): Number of iterations to run the sampler.
    """
    
    def __init__(self,  coarse_model: ModelProtocol, fine_model: ModelProtocol, kernel: KernelProtocol, proposal_coarse: ProposalProtocol, proposal_fine: ProposalProtocol, initial_position_coarse: np.ndarray, initial_position_fine: np.ndarray, n_iterations: int):

        dim = initial_position_coarse.shape[0]
        assert dim == initial_position_fine.shape[0], "The dimensions of the two chains must be the same."
        self.dim = dim

        self.kernel = kernel
        
        self.proposal_coarse = proposal_coarse
        self.proposal_fine = proposal_fine

        self.coarse_model = coarse_model
        self.fine_model = fine_model

        self.initial_state_coarse = ChainState(position=initial_position_coarse, **coarse_model(initial_position_coarse), metadata={
            'covariance': proposal_coarse.sigma if hasattr(proposal_coarse, 'sigma') else proposal_coarse.proposal.sigma,
            'mean': initial_position_coarse,
            'lambda': 2.4**2 / dim,
            'acceptance_probability': 0.0,
            'iteration': 1
            })
        
        self.initial_state_fine = ChainState(position=initial_position_fine, **fine_model(initial_position_fine), metadata={
            'covariance': proposal_fine.sigma if hasattr(proposal_fine, 'sigma') else proposal_fine.proposal.sigma,
            'mean': initial_position_fine,
            'lambda': 2.4**2 / dim,
            'acceptance_probability': 0.0,
            'iteration': 1
            })
        
        self.n_iterations = n_iterations

    def run(self, output_dir: str) -> None:

        """
        Run the coupled MCMC sampler for a specified number of iterations.
        
        Parameters:
        ----------
            output_dir (str): Directory to save the chain states.

        Returns:
        -------
            acceptance_rate_coarse (float): Acceptance rate for the coarse chain.
            acceptance_rate_fine (float): Acceptance rate for the fine chain.
            Also saves the samples to a file.
        """

        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the chain state
        current_coarse_state = self.initial_state_coarse
        current_fine_state = self.initial_state_fine

        samples_coarse = []
        samples_fine = []

        acceptance_count_coarse = 0
        acceptance_count_fine = 0

        # Run the coupled MCMC sampling loop
        for i in range(1, self.n_iterations+1):

            # Propose a new state for both chains
            proposed_coarse_state, proposed_fine_state = self.kernel.propose(self.proposal_coarse, self.proposal_fine, current_coarse_state, current_fine_state) # Metadata is copied from current_state

            # Compute the acceptance ratio
            ar_coarse, ar_fine = self.kernel.acceptance_ratio(self.proposal_coarse, current_coarse_state, proposed_coarse_state, self.proposal_fine, current_fine_state, proposed_fine_state)

            u = np.random.rand()

            # Accept or reject the proposed state
            if ar_coarse == 1 or u < ar_coarse:
                current_coarse_state = proposed_coarse_state
                acceptance_count_coarse += 1
            
            if ar_fine == 1 or u < ar_fine:
                current_fine_state = proposed_fine_state
                acceptance_count_fine += 1

            # Update the metadata for the proposed state
            current_coarse_state.metadata['iteration'] = i
            current_coarse_state.metadata['acceptance_probability'] = ar_coarse
            current_fine_state.metadata['iteration'] = i
            current_fine_state.metadata['acceptance_probability'] = ar_fine

            # Adapt the proposal distribution
            self.kernel.adapt(self.proposal_coarse, current_coarse_state, self.proposal_fine, current_fine_state)

            if hasattr(self.kernel, 'adapt_maps'):
                self.kernel.adapt_maps(samples_coarse, samples_fine)

            # Store the current state
            samples_coarse.append(copy.deepcopy(current_coarse_state))
            samples_fine.append(copy.deepcopy(current_fine_state))

        # Save the samples to a file
        with open(f"{output_dir}/samples_coarse.pkl", "wb") as f:
            pickle.dump(samples_coarse, f)

        with open(f"{output_dir}/samples_fine.pkl", "wb") as f:
            pickle.dump(samples_fine, f)
        
        # Save the acceptance rate
        acceptance_rate_coarse = acceptance_count_coarse / self.n_iterations
        acceptance_rate_fine = acceptance_count_fine / self.n_iterations
        return acceptance_rate_coarse, acceptance_rate_fine

    @staticmethod
    def load_samples(output_dir: str) -> Tuple[List[ChainState], List[ChainState]]:
        """
        Load MCMC samples from a pickle file.

        Parameters:
        ----------
            None

        Returns:
        -------
            samples (list): List of ChainState objects representing the MCMC samples.
        """
        with open(f'{output_dir}/samples_coarse.pkl', "rb") as f:
            samples_coarse = pickle.load(f)

        with open(f'{output_dir}/samples_fine.pkl', "rb") as f:
            samples_fine = pickle.load(f)

        return samples_coarse, samples_fine