"""
Coupled-chain MCMC sampler.

Orchestrates a coupled kernel (CoupledKernelBase) to run two chains (coarse and
fine) with marginal MH acceptance. The kernel holds the coupled proposal;
the sampler drives the loop and optional map adapt/save for transport kernels.
"""

from __future__ import annotations

import copy
import logging
import os
import pickle
from typing import List, Optional

import numpy as np

from samosa.core.state import ChainState
from samosa.core.kernel import CoupledKernelBase
from samosa.core.proposal import Proposal

logger = logging.getLogger(__name__)


class CoupledChainSampler:
    """
    Coupled-chain MCMC sampler.

    Runs two chains (coarse and fine) using a CoupledKernelBase kernel. The
    kernel owns the coupled proposal; the sampler drives the loop and manages
    state, metadata, and I/O. proposal_coarse and proposal_fine are used only
    for initial state metadata (e.g. covariance); kernel uses its
    coupled_proposal for propose/acceptance_ratio/adapt.

    Attributes
    ----------
    kernel : CoupledKernelBase
        The coupled transition kernel (holds coarse_model, fine_model, coupled_proposal).
    proposal_coarse : Proposal
        Coarse proposal (for initial state metadata).
    proposal_fine : Proposal
        Fine proposal (for initial state metadata).
    initial_state_coarse : ChainState
        Initial coarse state.
    initial_state_fine : ChainState
        Initial fine state.
    n_iterations : int
        Number of iterations.
    print_iteration : int, optional
        Print progress every N iterations.
    save_iteration : int, optional
        Save checkpoint every N iterations.
    """

    def __init__(
        self,
        kernel: CoupledKernelBase,
        proposal_coarse: Proposal,
        proposal_fine: Proposal,
        initial_position_coarse: np.ndarray,
        initial_position_fine: np.ndarray,
        n_iterations: int,
        print_iteration: Optional[int] = None,
        save_iteration: Optional[int] = None,
        restart_coarse: Optional[List[ChainState]] = None,
        restart_fine: Optional[List[ChainState]] = None,
    ) -> None:

        dim = initial_position_coarse.shape[0]
        if initial_position_fine.shape[0] != dim:
            raise ValueError("The dimensions of the two chains must be the same.")
        self.dim = dim

        self.kernel = kernel
        self.proposal_coarse = proposal_coarse
        self.proposal_fine = proposal_fine
        self.coarse_model = kernel.coarse_model
        self.fine_model = kernel.fine_model
        self.restart_coarse = restart_coarse
        self.restart_fine = restart_fine

        if self.restart_coarse is not None and self.restart_fine is not None:
            if len(self.restart_coarse) != len(self.restart_fine):
                raise ValueError(
                    "The lengths of restart_coarse and restart_fine must be the same."
                )

        if self.restart_coarse is not None:
            self.initial_state_coarse = self.restart_coarse[-1]
            meta = self.restart_coarse[-1].metadata or {}
            self.start_iteration = meta.get("iteration", 0) + 1
        else:
            self.initial_state_coarse = ChainState(
                position=initial_position_coarse,
                **self.coarse_model(initial_position_coarse),
                metadata={
                    "covariance": self.proposal_coarse.cov.copy(),
                    "mean": initial_position_coarse,
                    "lambda": 2.38**2 / self.dim,
                    "acceptance_probability": 0.0,
                    "is_accepted": False,
                    "iteration": 1,
                },
            )
            self.start_iteration = 1

        if self.restart_fine is not None:
            self.initial_state_fine = self.restart_fine[-1]
            meta = self.restart_fine[-1].metadata or {}
            self.start_iteration = meta.get("iteration", 0) + 1
        else:
            self.initial_state_fine = ChainState(
                position=initial_position_fine,
                **self.fine_model(initial_position_fine),
                metadata={
                    "covariance": self.proposal_fine.cov.copy(),
                    "mean": initial_position_fine,
                    "lambda": 2.38**2 / self.dim,
                    "acceptance_probability": 0.0,
                    "is_accepted": False,
                    "iteration": 1,
                },
            )

        self.n_iterations = n_iterations
        self.print_iteration = print_iteration
        self.save_iterations = save_iteration

    def run(self, output_dir: str) -> Optional[tuple[float, float]]:
        """
        Run the coupled MCMC sampler for a specified number of iterations.

        Parameters
        ----------
        output_dir : str
            Directory to save chain states and checkpoints.

        Returns
        -------
        tuple of (float, float), optional
            (acceptance_rate_coarse, acceptance_rate_fine). None if no iterations run.
        """
        os.makedirs(output_dir, exist_ok=True)

        current_coarse_state = self.initial_state_coarse
        current_fine_state = self.initial_state_fine

        if self.restart_coarse is not None and self.restart_fine is not None:
            samples_coarse = copy.deepcopy(self.restart_coarse)
            samples_fine = copy.deepcopy(self.restart_fine)
        else:
            samples_coarse: List[ChainState] = []
            samples_fine: List[ChainState] = []

        acceptance_count_coarse = 0
        acceptance_count_fine = 0

        logger.info(
            "Coupled-chain MCMC run starting: iterations=%s, start=%s",
            self.n_iterations,
            self.start_iteration,
        )

        for i in range(self.start_iteration, self.n_iterations + 1):
            if self.print_iteration is not None and i % self.print_iteration == 0:
                logger.info("Iteration %s/%s", i, self.n_iterations)

            proposed_coarse_state, proposed_fine_state = self.kernel.propose(
                current_coarse_state,
                current_fine_state,
            )

            ar_coarse, ar_fine = self.kernel.acceptance_ratio(
                current_coarse_state,
                proposed_coarse_state,
                current_fine_state,
                proposed_fine_state,
            )

            u = np.random.rand()

            if ar_coarse >= 1.0 or u < ar_coarse:
                current_coarse_state = proposed_coarse_state
                acceptance_count_coarse += 1
                is_accepted_coarse = True
            else:
                is_accepted_coarse = False

            if ar_fine >= 1.0 or u < ar_fine:
                current_fine_state = proposed_fine_state
                acceptance_count_fine += 1
                is_accepted_fine = True
            else:
                is_accepted_fine = False

            if current_coarse_state.metadata is None:
                current_coarse_state.metadata = {}
            if current_fine_state.metadata is None:
                current_fine_state.metadata = {}
            current_coarse_state.metadata["iteration"] = i
            current_coarse_state.metadata["acceptance_probability"] = ar_coarse
            current_coarse_state.metadata["is_accepted"] = is_accepted_coarse
            current_fine_state.metadata["iteration"] = i
            current_fine_state.metadata["acceptance_probability"] = ar_fine
            current_fine_state.metadata["is_accepted"] = is_accepted_fine

            samples_coarse.append(copy.deepcopy(current_coarse_state))
            samples_fine.append(copy.deepcopy(current_fine_state))

            self.kernel.adapt(
                current_coarse_state,
                current_fine_state,
                samples=(samples_coarse, samples_fine),
            )

            if self.save_iterations is not None and i % self.save_iterations == 0:
                self._save_checkpoint(output_dir, i, samples_coarse, samples_fine)

        self._save_checkpoint(
            output_dir,
            self.n_iterations,
            samples_coarse,
            samples_fine,
            final_checkpoint=True,
        )

        n_run = self.n_iterations - self.start_iteration + 1
        if n_run > 0:
            acceptance_rate_coarse = acceptance_count_coarse / n_run
            acceptance_rate_fine = acceptance_count_fine / n_run
        else:
            acceptance_rate_coarse = 0.0
            acceptance_rate_fine = 0.0

        logger.info(
            "Coupled-chain MCMC run finished: acceptance_rate_coarse=%.4f, acceptance_rate_fine=%.4f",
            acceptance_rate_coarse,
            acceptance_rate_fine,
        )
        return acceptance_rate_coarse, acceptance_rate_fine

    def _save_checkpoint(
        self,
        output_dir: str,
        iteration: int,
        samples_coarse: List[ChainState],
        samples_fine: List[ChainState],
        *,
        final_checkpoint: bool = False,
    ) -> None:
        """
        Save a checkpoint: coarse/fine samples and optionally kernel transport maps.

        Uses the same iteration number for samples and maps so checkpoints stay in sync.
        Intermediate checkpoints go to output_dir/samples/ and output_dir/maps/;
        final checkpoint writes samples to output_dir root.
        """
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/maps", exist_ok=True)
        if final_checkpoint:
            path_c = os.path.join(output_dir, "samples_coarse.pkl")
            path_f = os.path.join(output_dir, "samples_fine.pkl")
        else:
            path_c = os.path.join(
                output_dir, "samples", f"samples_coarse_{iteration}.pkl"
            )
            path_f = os.path.join(
                output_dir, "samples", f"samples_fine_{iteration}.pkl"
            )
        with open(path_c, "wb") as f:
            pickle.dump(samples_coarse, f)
        with open(path_f, "wb") as f:
            pickle.dump(samples_fine, f)
        logger.info(
            "Checkpoint saved: iteration=%s, paths=%s, %s", iteration, path_c, path_f
        )

        save_map_coarse = getattr(self.proposal_coarse, "save_map", None)
        save_map_fine = getattr(self.proposal_fine, "save_map", None)
        if save_map_coarse is not None:
            map_dir_coarse = (
                output_dir if final_checkpoint else f"{output_dir}/maps/map_coarse"
            )
            try:
                save_map_coarse(map_dir_coarse, iteration)
                logger.debug("Coarse map checkpoint saved: iteration=%s", iteration)
            except Exception as e:
                logger.warning(
                    "Coarse map checkpoint failed at iteration %s: %s", iteration, e
                )
        if save_map_fine is not None:
            map_dir_fine = (
                output_dir if final_checkpoint else f"{output_dir}/maps/map_fine"
            )
            try:
                save_map_fine(map_dir_fine, iteration)
                logger.debug("Fine map checkpoint saved: iteration=%s", iteration)
            except Exception as e:
                logger.warning(
                    "Fine map checkpoint failed at iteration %s: %s", iteration, e
                )


# Backward compatibility
coupledMCMCsampler = CoupledChainSampler
