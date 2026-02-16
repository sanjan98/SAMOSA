"""
Coupled-chain MCMC sampler.

Orchestrates a coupled kernel and two proposals to run two chains
(coarse and fine) with the same marginal MH rule and optional map adapt/save.
"""

import copy
import os
import pickle
from typing import List, Optional

import numpy as np

from samosa.core.state import ChainState
from samosa.core.kernel import CoupledKernelProtocol
from samosa.core.proposal import ProposalBase


class CoupledChainSampler:
    """
    Class for a coupled chain MCMC sampler.

    Attributes:
        kernel (TransitionKernel): The transition kernel used for sampling.
        proposal (Proposal): The proposal distribution used for generating candidate states.
        model (BaseModel): The model used for computing posterior values.
        initial_state (ChainState): The initial state of the chain.
        n_iterations (int): Number of iterations to run the sampler.
    """

    def __init__(
        self,
        kernel: CoupledKernelProtocol,
        proposal_coarse: ProposalBase,
        proposal_fine: ProposalBase,
        initial_position_coarse: np.ndarray,
        initial_position_fine: np.ndarray,
        n_iterations: int,
        print_iteration: Optional[int] = None,
        save_iteration: Optional[int] = None,
        restart_coarse: Optional[List[ChainState]] = None,
        restart_fine: Optional[List[ChainState]] = None,
    ) -> None:

        dim = initial_position_coarse.shape[0]
        assert dim == initial_position_fine.shape[0], (
            "The dimensions of the two chains must be the same."
        )
        self.dim = dim

        self.kernel = kernel

        self.proposal_coarse = proposal_coarse
        self.proposal_fine = proposal_fine

        self.coarse_model = kernel.coarse_model
        self.fine_model = kernel.fine_model

        self.restart_coarse = restart_coarse
        self.restart_fine = restart_fine

        if self.restart_coarse is not None and self.restart_fine is not None:
            assert len(self.restart_coarse) == len(self.restart_fine), (
                "The lengths of restart_coarse and restart_fine must be the same."
            )

        if self.restart_coarse is not None:
            self.initial_state_coarse = self.restart_coarse[-1]
            self.start_iteration = self.restart_coarse[-1].metadata["iteration"] + 1

            # Add a check here to reset mean and covariance for the reference space
            if self.initial_state_coarse.reference_position is None and hasattr(
                self.kernel, "coarse_map"
            ):
                self.initial_state_coarse.metadata["mean"] = (
                    self.kernel.coarse_map.forward(self.initial_state_coarse.position)[
                        0
                    ]
                )
                self.initial_state_coarse.metadata["covariance"] = (
                    proposal_coarse.sigma
                    if hasattr(proposal_coarse, "sigma")
                    else proposal_coarse.proposal.sigma
                )
                self.initial_state_coarse.metadata["lambda"] = 2.4**2 / dim

        else:
            self.initial_state_coarse = ChainState(
                position=initial_position_coarse,
                **self.coarse_model(initial_position_coarse),
                metadata={
                    "covariance": proposal_coarse.sigma
                    if hasattr(proposal_coarse, "sigma")
                    else proposal_coarse.proposal.sigma,
                    "mean": self.kernel.coarse_map.forward(initial_position_coarse)[0]
                    if hasattr(self.kernel, "coarse_map")
                    else initial_position_coarse,
                    "lambda": 2.4**2 / dim,
                    "acceptance_probability": 0.0,
                    "is_accepted": False,
                    "iteration": 1,
                },
            )
            self.start_iteration = 1

        if self.restart_fine is not None:
            self.initial_state_fine = self.restart_fine[-1]

            # Add a check here to reset mean and covariance for the reference space
            if self.initial_state_fine.reference_position is None and hasattr(
                self.kernel, "fine_map"
            ):
                self.initial_state_fine.metadata["mean"] = self.kernel.fine_map.forward(
                    self.initial_state_fine.position
                )[0]
                self.initial_state_fine.metadata["covariance"] = (
                    proposal_fine.sigma
                    if hasattr(proposal_fine, "sigma")
                    else proposal_fine.proposal.sigma
                )
                self.initial_state_fine.metadata["lambda"] = 2.4**2 / dim

        else:
            self.initial_state_fine = ChainState(
                position=initial_position_fine,
                **self.fine_model(initial_position_fine),
                metadata={
                    "covariance": proposal_fine.sigma
                    if hasattr(proposal_fine, "sigma")
                    else proposal_fine.proposal.sigma,
                    "mean": self.kernel.fine_map.forward(initial_position_fine)[0]
                    if hasattr(self.kernel, "fine_map")
                    else initial_position_fine,
                    "lambda": 2.4**2 / dim,
                    "acceptance_probability": 0.0,
                    "is_accepted": False,
                    "iteration": 1,
                },
            )

        self.n_iterations = n_iterations
        self.print_iteration = print_iteration
        self.save_iterations = save_iteration

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

        if self.restart_coarse is not None and self.restart_fine is not None:
            samples_coarse = copy.deepcopy(self.restart_coarse)
            samples_fine = copy.deepcopy(self.restart_fine)
        else:
            samples_coarse = []
            samples_fine = []

        acceptance_count_coarse = 0
        acceptance_count_fine = 0

        # Run the coupled MCMC sampling loop
        for i in range(self.start_iteration, self.n_iterations + 1):
            if self.print_iteration is not None and i % self.print_iteration == 0:
                print(f"Iteration {i}/{self.n_iterations}")

            # Propose a new state for both chains
            (
                proposed_coarse_state,
                proposed_fine_state,
                current_coarse_state,
                current_fine_state,
            ) = self.kernel.propose(
                self.proposal_coarse,
                self.proposal_fine,
                current_coarse_state,
                current_fine_state,
            )  # Metadata is copied from current_state

            # Compute the acceptance ratio
            ar_coarse, ar_fine = self.kernel.acceptance_ratio(
                self.proposal_coarse,
                current_coarse_state,
                proposed_coarse_state,
                self.proposal_fine,
                current_fine_state,
                proposed_fine_state,
            )

            u = np.random.rand()

            # Accept or reject the proposed state
            if ar_coarse == 1 or u < ar_coarse:
                current_coarse_state = proposed_coarse_state
                acceptance_count_coarse += 1
                is_accepted_coarse = True
            else:
                is_accepted_coarse = False

            if ar_fine == 1 or u < ar_fine:
                current_fine_state = proposed_fine_state
                acceptance_count_fine += 1
                is_accepted_fine = True
            else:
                is_accepted_fine = False

            # Update the metadata for the proposed state
            current_coarse_state.metadata["iteration"] = i
            current_coarse_state.metadata["acceptance_probability"] = ar_coarse
            current_coarse_state.metadata["is_accepted"] = is_accepted_coarse
            current_fine_state.metadata["iteration"] = i
            current_fine_state.metadata["acceptance_probability"] = ar_fine
            current_fine_state.metadata["is_accepted"] = is_accepted_fine

            # Adapt the proposal distribution
            self.kernel.adapt(
                self.proposal_coarse,
                current_coarse_state,
                self.proposal_fine,
                current_fine_state,
            )

            # Store the current state
            samples_coarse.append(copy.deepcopy(current_coarse_state))
            samples_fine.append(copy.deepcopy(current_fine_state))

            if hasattr(self.kernel, "adapt_maps"):
                self.kernel.adapt_maps(samples_coarse, samples_fine)

            # Save the samples at specified intervals
            if self.save_iterations is not None and i % self.save_iterations == 0:
                with open(f"{output_dir}/samples_coarse{i}.pkl", "wb") as f:
                    pickle.dump(samples_coarse, f)
                with open(f"{output_dir}/samples_fine{i}.pkl", "wb") as f:
                    pickle.dump(samples_fine, f)
                print(f"Saved samples at iteration {i} to {output_dir}/samples_{i}.pkl")

                # Save the transport map if applicable
                if hasattr(self.kernel, "save_maps"):
                    self.kernel.save_maps(output_dir, i)

        # Save the samples to a file
        with open(f"{output_dir}/samples_coarse.pkl", "wb") as f:
            pickle.dump(samples_coarse, f)

        with open(f"{output_dir}/samples_fine.pkl", "wb") as f:
            pickle.dump(samples_fine, f)

        # Save the final map if applicable
        if hasattr(self.kernel, "save_maps"):
            self.kernel.save_maps(output_dir, self.n_iterations)

        # Save the acceptance rate
        if self.start_iteration > 1:
            acceptance_rate_coarse = acceptance_count_coarse / (
                self.n_iterations - self.start_iteration + 1
            )
            acceptance_rate_fine = acceptance_count_fine / (
                self.n_iterations - self.start_iteration + 1
            )
        else:
            acceptance_rate_coarse = acceptance_count_coarse / self.n_iterations
            acceptance_rate_fine = acceptance_count_fine / self.n_iterations
        return acceptance_rate_coarse, acceptance_rate_fine


# Backward compatibility
coupledMCMCsampler = CoupledChainSampler
