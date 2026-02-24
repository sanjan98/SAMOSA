"""
Single-chain MCMC sampler.

Orchestrates a single-fidelity kernel to run one chain; supports MH-style
and delayed-rejection kernels. The kernel owns model and proposal; the
sampler drives the loop (propose -> accept/reject -> adapt) and uses the
kernel's proposal for initial metadata and optional map saving.
"""

from __future__ import annotations

import copy
import logging
import os
import pickle
from dataclasses import replace
from typing import List, Optional

import numpy as np

from samosa.core.state import ChainState
from samosa.kernels.metropolis import MetropolisHastingsKernel
from samosa.kernels.delayedrejection import DelayedRejectionKernel

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # or DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class SingleChainSampler:
    """
    Single-chain MCMC sampler.

    Runs one chain using a transition kernel (Metropolis-Hastings or Delayed
    Rejection). The kernel owns the model and proposal; the sampler drives
    the loop and manages state, metadata, and I/O.

    Attributes
    ----------
    kernel : MetropolisHastingsKernel | DelayedRejectionKernel
        The transition kernel (holds model and proposal).
    initial_state : ChainState
        Initial state of the chain.
    n_iterations : int
        Number of iterations.
    print_iteration : int, optional
        Print progress every N iterations.
    save_iteration : int, optional
        Save checkpoint every N iterations.
    restart : list of ChainState, optional
        States to restart from (uses last as initial state). Iteration and
        adaptation (e.g. Haario mean/covariance) continue from the last
        state's metadata: start_iteration = last.metadata["iteration"] + 1.
        If you pass a subset (e.g. last 2000 of 10000), keep the original
        iteration and metadata on the last state so adaptation continues
        correctly. Do not reset iteration to 1..len(restart) unless you
        also recompute mean/covariance from the subset and set them in
        the last state's metadata.
    """

    def __init__(
        self,
        kernel: MetropolisHastingsKernel | DelayedRejectionKernel,
        initial_position: np.ndarray,
        n_iterations: int,
        print_iteration: Optional[int] = None,
        save_iteration: Optional[int] = None,
        restart: Optional[List[ChainState]] = None,
    ) -> None:
        dim = initial_position.shape[0]
        self.dim = dim
        self.kernel = kernel
        self.model = kernel.model
        proposal = kernel.proposal
        self.restart = restart

        if self.restart is not None:
            initial = self.restart[-1]
            # When using a transport proposal, ensure restarted state has reference_position
            tmap = getattr(proposal, "map", None)
            if tmap is not None and initial.reference_position is None:
                ref_pos, _ = tmap.forward(initial.position)
                self.initial_state = replace(initial, reference_position=ref_pos)
            else:
                self.initial_state = initial
            meta = self.initial_state.metadata or {}
            self.start_iteration = meta.get("iteration", 0) + 1
        else:
            # For transport proposals, set reference_position so all states have it
            ref_pos: Optional[np.ndarray] = None
            tmap = getattr(proposal, "map", None)
            if tmap is not None:
                ref_pos, _ = tmap.forward(initial_position)
            self.initial_state = ChainState(
                position=initial_position,
                reference_position=ref_pos,
                **self.model(initial_position),
                metadata={
                    "covariance": proposal.cov.copy(),
                    "mean": initial_position,
                    "lambda": 2.38**2 / dim,
                    "acceptance_probability": 0.0,
                    "is_accepted": False,
                    "iteration": 1,
                },
            )
            self.start_iteration = 1

        self.n_iterations = n_iterations
        self.print_iteration = print_iteration
        self.save_iterations = save_iteration
        self.is_delayed_rejection = isinstance(kernel, DelayedRejectionKernel)

    def run(self, output_dir: str) -> Optional[float]:
        """
        Run the MCMC sampler for a specified number of iterations.

        Args:
        output_dir : str
            Directory to save chain states and checkpoints.

        Returns:
        float, optional
            Acceptance rate over the run. None if no iterations run.
        """
        os.makedirs(output_dir, exist_ok=True)
        current_state = self.initial_state
        if self.restart is not None:
            samples = copy.deepcopy(self.restart)
        else:
            samples: List[ChainState] = []
        acceptance_count = 0

        logger.info(
            "Single-chain MCMC run starting: iterations=%s, start=%s",
            self.n_iterations,
            self.start_iteration,
        )

        for i in range(self.start_iteration, self.n_iterations + 1):
            if self.print_iteration is not None and i % self.print_iteration == 0:
                logger.info("Iteration %s/%s", i, self.n_iterations)

            if self.is_delayed_rejection:
                # Delayed Rejection: propose() returns accepted state or current
                proposed_state = self.kernel.propose(current_state)
                if proposed_state is not current_state:
                    acceptance_count += 1
                    is_accepted = True
                else:
                    is_accepted = False
                ar = getattr(self.kernel, "ar", 0.0)
                current_state = proposed_state
            else:
                # Metropolis-Hastings: propose then accept/reject
                proposed_state = self.kernel.propose(current_state)
                ar = self.kernel.acceptance_ratio(current_state, proposed_state)
                if ar >= 1.0 or np.random.rand() < ar:
                    current_state = proposed_state
                    acceptance_count += 1
                    is_accepted = True
                else:
                    is_accepted = False

            if current_state.metadata is None:
                current_state.metadata = {}
            current_state.metadata["iteration"] = i
            current_state.metadata["acceptance_probability"] = ar
            current_state.metadata["is_accepted"] = is_accepted

            samples.append(copy.deepcopy(current_state))

            self.kernel.adapt(current_state, samples=samples)

            if self.save_iterations is not None and i % self.save_iterations == 0:
                self._save_checkpoint(output_dir, i, samples)

        self._save_checkpoint(
            output_dir, self.n_iterations, samples, final_checkpoint=True
        )

        n_run = self.n_iterations - self.start_iteration + 1
        acceptance_rate = acceptance_count / n_run if n_run else 0.0
        logger.info(
            "Single-chain MCMC run finished: acceptance_rate=%.4f", acceptance_rate
        )
        return acceptance_rate

    def _save_checkpoint(
        self,
        output_dir: str,
        iteration: int,
        samples: List[ChainState],
        final_checkpoint: bool = False,
    ) -> None:
        """
        Save a checkpoint: samples to pickle and optionally the kernel's transport map.

        Uses the same iteration number for both so sample and map checkpoints stay in sync.
        No separate counter is needed; the MCMC iteration is the checkpoint identifier.
        """
        if final_checkpoint:
            path = f"{output_dir}/samples.pkl"
        else:
            os.makedirs(f"{output_dir}/samples", exist_ok=True)
            path = f"{output_dir}/samples/samples_{iteration}.pkl"
        with open(path, "wb") as f:
            pickle.dump(samples, f)
        logger.info("Checkpoint saved: iteration=%s, path=%s", iteration, path)
        save_map = getattr(self.kernel.proposal, "save_map", None)
        if save_map is not None:
            # Proposal.save_map(output_dir, iteration) builds path as output_dir/map_{iteration}
            if final_checkpoint:
                map_dir = output_dir
            else:
                os.makedirs(f"{output_dir}/maps", exist_ok=True)
                map_dir = f"{output_dir}/maps"
            try:
                save_map(map_dir, iteration)
                logger.debug("Map checkpoint saved: iteration=%s", iteration)
            except Exception as e:
                logger.warning(
                    "Map checkpoint failed at iteration %s: %s", iteration, e
                )


# Backward compatibility
MCMCsampler = SingleChainSampler
