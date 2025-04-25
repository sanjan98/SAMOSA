"""
Class file for a single chain MCMC sampler.
"""

from samosa.core.model import ModelProtocol
from samosa.core.state import ChainState
from samosa.core.kernel import KernelProtocol
from samosa.core.proposal import ProposalProtocol
from samosa.kernels.delayedrejection import DelayedRejectionKernel
from typing import List
import numpy as np
import os
import copy

import pickle

class MCMCsampler:
    """
    Class for a single chain MCMC sampler.
    
    Attributes:
        kernel (TransitionKernel): The transition kernel used for sampling.
        proposal (Proposal): The proposal distribution used for generating candidate states.
        model (BaseModel): The model used for computing posterior values.
        initial_state (ChainState): The initial state of the chain.
        n_iterations (int): Number of iterations to run the sampler.
    """
    
    def __init__(self,  model: ModelProtocol, kernel: KernelProtocol, proposal: ProposalProtocol, initial_position: np.ndarray, n_iterations: int):

        dim = initial_position.shape[0]
        self.dim = dim
        self.kernel = kernel
        self.proposal = proposal
        self.model = model
        self.initial_state = ChainState(position=initial_position, **model(initial_position), metadata={
            'covariance': proposal.sigma if hasattr(proposal, 'sigma') else proposal.proposal.sigma,
            'mean': initial_position,
            'lambda': 2.4**2 / dim,
            'acceptance_probability': 0.0,
            'iteration': 1
            })
        self.n_iterations = n_iterations
        self.is_delayed_rejection = isinstance(kernel, DelayedRejectionKernel)

    def run(self, output_dir: str) -> None:

        """
        Run the MCMC sampler for a specified number of iterations.
        
        Parameters:
        ----------
            output_dir (str): Directory to save the chain states.

        Returns:
        -------
            None, but saves the chain states to a file.
        """

        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the chain state
        current_state = self.initial_state
        samples = []
        acceptance_count = 0

        # Run the MCMC sampling loop
        for i in range(1, self.n_iterations+1):

            # Check for delayed rejection kernel
            if self.is_delayed_rejection:
                proposed_state = self.kernel.propose(self.proposal, current_state)

                # Check if state was changed
                if proposed_state is not current_state:
                    acceptance_count += 1

                ar = getattr(self.kernel, 'ar', 0.0)

                # Update the next state
                current_state = proposed_state

            # Matropolis-Hastings kernel
            else:

                # Propose a new state
                proposed_state = self.kernel.propose(self.proposal, current_state) # Metadata is copied from current_state

                # Compute the acceptance ratio
                ar = self.kernel.acceptance_ratio(self.proposal, current_state, proposed_state)

                # Accept or reject the proposed state
                if ar == 1 or np.random.rand() < ar:
                    current_state = proposed_state
                    acceptance_count += 1

            # Update the metadata for the proposed state
            current_state.metadata['iteration'] = i
            current_state.metadata['acceptance_probability'] = ar

            # Adapt the proposal distribution
            self.kernel.adapt(self.proposal, current_state)

            # Store the current state
            samples.append(copy.deepcopy(current_state))

        # Save the samples to a file
        with open(f"{output_dir}/samples.pkl", "wb") as f:
            pickle.dump(samples, f)

    @staticmethod
    def load_samples(output_dir: str) -> List[ChainState]:
        """
        Load MCMC samples from a pickle file.

        Parameters:
        ----------
            None

        Returns:
        -------
            samples (list): List of ChainState objects representing the MCMC samples.
        """
        with open(f'{output_dir}/samples.pkl', "rb") as f:
            samples = pickle.load(f)
            return samples