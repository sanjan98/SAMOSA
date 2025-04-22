"""
Class file for a single chain MCMC sampler.
"""

from core.model import BaseModel
from core.state import ChainState
from core.kernel import TransitionKernel
from core.proposal import Proposal
from typing import Any, Dict
import numpy as np

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
    
    def __init__(self,  model: BaseModel, kernel: TransitionKernel, proposal: Proposal, initial_position: np.ndarray, n_iterations: int):

        dim = initial_position.shape[0]
        self.dim = dim
        self.kernel = kernel
        self.proposal = proposal
        self.model = model
        self.initial_state = ChainState(position=initial_position, **model(initial_position), metadata={
            'covariance': proposal.sigma,
            'mean': initial_position,
            'lambda': 2.4**2 / dim,
            'acceptance_probability': 0.0,
            'iteration': 1
            })
        self.n_iterations = n_iterations

    def run(self, output_dir: str) -> Dict[str, Any]:

        """
        Run the MCMC sampler for a specified number of iterations.
        
        Returns:
            Dict[str, Any]: A dictionary containing the sampled states and other relevant information.
        """
        # Initialize the chain state
        current_state = self.initial_state
        samples = []

        # Run the MCMC sampling loop
        for i in range(1, self.n_iterations):
            # Propose a new state
            proposed_state = self.kernel.propose(self.proposal, current_state) # Metadata is copied from current_state

            # Compute the acceptance ratio
            ar = self.kernel.acceptance_ratio(self.proposal, current_state, proposed_state)

            # Accept or reject the proposed state
            if np.random.rand() < ar:
                current_state = proposed_state

            # Update the metadata for the proposed state
            proposed_state.metadata['iteration'] = i
            proposed_state.metadata['acceptance_probability'] = ar

            # Adapt the proposal distribution
            self.kernel.adapt(self.proposal, proposed_state)

            # Store the current state
            samples.append(current_state)

        # Save the samples to a file
        with open(f"{output_dir}/samples.pkl", "wb") as f:
            pickle.dump(samples, f)

        return None