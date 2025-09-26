"""
Simple Multi-Level Monte Carlo wrapper.
This wrapper manages multiple fidelity levels and distributes them across MPI processes.
"""

import os
import pickle
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from mpi4py import MPI

from samosa.core.kernel import KernelProtocol
from samosa.core.model import ModelProtocol
from samosa.core.proposal import ProposalProtocol
from samosa.core.state import ChainState
from samosa.samplers.coupled_chain import coupledMCMCsampler
from samosa.samplers.single_chain import MCMCsampler


@dataclass
class MLMCLevel:
    """Configuration for a single MLMC level"""

    level: int

    coarse_model: Optional[ModelProtocol]  # None for level 0
    fine_model: ModelProtocol

    coarse_proposal: Optional[ProposalProtocol]  # None for level 0
    fine_proposal: ProposalProtocol

    kernel: KernelProtocol
    n_samples: int

    initial_position_coarse: Optional[np.ndarray] = None  # None for level 0
    initial_position_fine: Optional[np.ndarray] = None

    restart_coarse: Optional[List[ChainState]] = None
    restart_fine: Optional[List[ChainState]] = None


@dataclass
class MLMCResults:
    """Results from MLMC estimation"""

    level_means: List[np.ndarray]  # Mean difference for each level
    level_variances: List[np.ndarray]  # Variance for each level

    mlmc_expectation: np.ndarray  # Final MLMC estimator
    mlmc_variance: np.ndarray  # MLMC estimator variance

    samples_per_level: List[int]
    computation_times: List[float]

    levels_completed: List[int]


class SimpleMLMCWrapper:
    """
    Simple Multi-Level Monte Carlo wrapper for coupled MCMC sampling.
    Distributes different levels across MPI processes without adaptive allocation.
    """

    def __init__(self, levels: List[MLMCLevel], output_dir: Optional[str] = "./mlmc_output", print_progress: Optional[bool] = True):
        """
        Initialize simple MLMC wrapper.

        Args:
            levels: List of MLMC level configurations
            output_dir: Directory for output files
            print_progress: Whether to print progress information
        """

        self.levels = sorted(levels, key=lambda x: x.level)
        self.output_dir = output_dir
        self.print_progress = print_progress

        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Validate levels
        self._validate_levels()

        # Create output directory
        if self.rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            if self.print_progress:
                print(f"MLMC Setup: {len(self.levels)} levels, {self.size} MPI processes")
        self.comm.Barrier()

    def run(self) -> Optional[MLMCResults]:
        """
        Run the ML-MCMC sampling routine with specified samples per level and specified coupling kernel.

        Returns:
            MLMCResults object on rank 0, None on other ranks
        """
        if self.rank == 0 and self.print_progress:
            print("Starting MLMC sampling...")
            print(f"Total samples: {[level.n_samples for level in self.levels]}")

        # Distribute levels across processes
        my_levels = self._distribute_levels()

        # Sample assigned levels
        local_results = {}

        for level_idx in my_levels:
            level = self.levels[level_idx]

            if self.print_progress:
                print(f"Process {self.rank}: Starting level {level.level}")

            samples_coarse, samples_fine, comp_time = self._sample_level(level)
            mean_diff, variance = self._compute_qoi_differences(samples_coarse, samples_fine)

            local_results[level_idx] = {
                "level": level.level,
                "mean": mean_diff,
                "variance": variance,
                "n_samples": level.n_samples,
                "computation_time": comp_time,
            }

            if self.print_progress:
                print(f"Process {self.rank}: Completed level {level.level} in {comp_time:.2f}s")

        # Gather results from all processes
        all_results = self.comm.gather(local_results, root=0)

        # Combine and return results on root process
        if self.rank == 0:
            # Combine results from all processes
            combined_results = {}
            for proc_results in all_results:
                combined_results.update(proc_results)

            # Sort by level
            sorted_levels = sorted(combined_results.keys())

            # Extract arrays
            level_means = [combined_results[i]["mean"] for i in sorted_levels]
            level_variances = [combined_results[i]["variance"] for i in sorted_levels]
            samples_per_level = [combined_results[i]["n_samples"] for i in sorted_levels]
            computation_times = [combined_results[i]["computation_time"] for i in sorted_levels]
            levels_completed = [combined_results[i]["level"] for i in sorted_levels]

            # Compute MLMC estimator
            mlmc_expectation = sum(level_means)
            mlmc_variance = sum(var / n_samples for var, n_samples in zip(level_variances, samples_per_level))

            # Create results object
            results = MLMCResults(
                level_means=level_means,
                level_variances=level_variances,
                mlmc_expectation=mlmc_expectation,
                mlmc_variance=mlmc_variance,
                samples_per_level=samples_per_level,
                computation_times=computation_times,
                levels_completed=levels_completed,
            )

            # Save results
            with open(f"{self.output_dir}/mlmc_results.pkl", "wb") as f:
                pickle.dump(results, f)

            if self.print_progress:
                print("\n" + "=" * 50)
                print("MLMC SAMPLING COMPLETE")
                print("=" * 50)
                print(f"Levels completed: {levels_completed}")
                print(f"Samples per level: {samples_per_level}")
                print(f"Total computation time: {sum(computation_times):.2f}s")
                print(f"MLMC expectation shape: {mlmc_expectation.shape}")
                print(f"MLMC variance shape: {mlmc_variance.shape}")
                print(f"Results saved to: {self.output_dir}/mlmc_results.pkl")
                print("=" * 50)

            return results

        return None

    def load_results(self) -> MLMCResults:
        """Load results from saved file"""
        with open(f"{self.output_dir}/mlmc_results.pkl", "rb") as f:
            return pickle.load(f)
        
    def _validate_levels(self):
        """Validate the MLMC level configuration"""
        # Check that level 0 has no coarse model
        if self.levels[0].level == 0:
            if self.levels[0].coarse_model is not None or self.levels[0].coarse_proposal is not None:
                raise ValueError("Level 0 should not have coarse model or proposal")

        # Check that levels > 0 have both coarse and fine models
        for level in self.levels[1:]:
            if level.coarse_model is None or level.coarse_proposal is None:
                raise ValueError(f"Level {level.level} must have both coarse and fine models/proposals")

        # Check level numbering
        expected_levels = list(range(len(self.levels)))
        actual_levels = [level.level for level in self.levels]
        if actual_levels != expected_levels:
            raise ValueError(f"Levels must be numbered 0, 1, 2, ... Got {actual_levels}")

    def _compute_qoi_differences(self, samples_coarse: List[ChainState], samples_fine: List[ChainState]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute QoI differences and statistics for a level"""
        n_samples = len(samples_fine)

        differences = []
        for i in range(n_samples):
            qoi_fine = self.qoi_function(samples_fine[i])

            if samples_coarse[i] is not None:  # Level > 0
                qoi_coarse = self.qoi_function(samples_coarse[i])
                diff = qoi_fine - qoi_coarse
            else:  # Level 0
                diff = qoi_fine

            differences.append(diff)

        differences = np.array(differences)
        mean_diff = np.mean(differences, axis=0)
        variance = np.var(differences, axis=0, ddof=1) if n_samples > 1 else np.zeros_like(mean_diff)

        return mean_diff, variance

    def _sample_level(self, level: MLMCLevel) -> Tuple[List[ChainState], List[ChainState], float]:
        """Sample a single MLMC level and return samples + computation time"""
        start_time = time.time()

        level_output_dir = f"{self.output_dir}/level_{level.level}_rank_{self.rank}"

        if level.level == 0:
            # Level 0: single chain sampling of finest model only
            if self.rank == 0 and self.print_progress:
                print(f"Sampling Level 0: {level.n_samples} samples (fine model only)")

            sampler = MCMCsampler(
                kernel=level.kernel,
                proposal=level.fine_proposal,
                initial_position=level.initial_position_fine,
                n_iterations=level.n_samples,
                print_iteration=max(1, level.n_samples // 10),
                save_iteraton=max(1000, level.n_samples),
                restart=level.restart_fine,
            )

            _ = sampler.run(level_output_dir)

            # Load samples
            with open(f"{level_output_dir}/samples.pkl", "rb") as f:
                samples_fine = pickle.load(f)

            # No coarse samples for level 0
            samples_coarse = [None] * len(samples_fine)

        else:
            # Level > 0: coupled chain sampling
            if self.rank == 0 and self.print_progress:
                print(f"Sampling Level {level.level}: {level.n_samples} samples (coupled chains)")

            sampler = coupledMCMCsampler(
                kernel=level.kernel,
                proposal_coarse=level.coarse_proposal,
                proposal_fine=level.fine_proposal,
                initial_position_coarse=level.initial_position_coarse,
                initial_position_fine=level.initial_position_fine,
                n_iterations=level.n_samples,
                print_iteration=max(1, level.n_samples // 10),
                save_iteration=max(1000, level.n_samples),
                restart_coarse=level.restart_coarse,
                restart_fine=level.restart_fine,
            )

            _ = sampler.run(level_output_dir)

            # Load samples
            with open(f"{level_output_dir}/samples_coarse.pkl", "rb") as f:
                samples_coarse = pickle.load(f)
            with open(f"{level_output_dir}/samples_fine.pkl", "rb") as f:
                samples_fine = pickle.load(f)

        end_time = time.time()
        computation_time = end_time - start_time

        return samples_coarse, samples_fine, computation_time

    def _distribute_levels(self) -> List[int]:
        """Distribute levels across MPI processes"""
        levels_per_process = len(self.levels) // self.size
        remainder = len(self.levels) % self.size

        # Give extra levels to first 'remainder' processes
        if self.rank < remainder:
            start_level = self.rank * (levels_per_process + 1)
            n_levels = levels_per_process + 1
        else:
            start_level = self.rank * levels_per_process + remainder
            n_levels = levels_per_process

        my_levels = list(range(start_level, start_level + n_levels))

        if self.rank == 0 and self.print_progress:
            print("Level distribution across MPI processes:")
            for rank in range(self.size):
                if rank < remainder:
                    rank_start = rank * (levels_per_process + 1)
                    rank_n = levels_per_process + 1
                else:
                    rank_start = rank * levels_per_process + remainder
                    rank_n = levels_per_process
                rank_levels = list(range(rank_start, rank_start + rank_n))
                print(f"  Process {rank}: levels {rank_levels}")

        return my_levels


# Utility function for easy setup
def create_mlmc_levels(
    models: List[Tuple[ModelProtocol, ModelProtocol]],
    proposals: List[Tuple[ProposalProtocol, ProposalProtocol]],
    kernels: List[KernelProtocol],
    samples_per_level: List[int],
    initial_positions: List[Tuple[np.ndarray, np.ndarray]],
) -> List[MLMCLevel]:
    """
    Utility function to create MLMC levels from lists of components.

    Args:
        models: List of (coarse_model, fine_model) tuples. Use (None, fine_model) for level 0.
        proposals: List of (coarse_proposal, fine_proposal) tuples. Use (None, fine_proposal) for level 0.
        kernels: List of kernel objects for each level
        samples_per_level: Number of samples for each level
        initial_positions: List of (coarse_initial, fine_initial) tuples. Use (None, fine_initial) for level 0.

    Returns:
        List of MLMCLevel objects
    """
    if not (len(models) == len(proposals) == len(kernels) == len(samples_per_level) == len(initial_positions)):
        raise ValueError("All input lists must have the same length")

    levels = []
    for i, (
        (coarse_model, fine_model),
        (coarse_proposal, fine_proposal),
        kernel,
        n_samples,
        (coarse_init, fine_init),
    ) in enumerate(zip(models, proposals, kernels, samples_per_level, initial_positions)):
        level = MLMCLevel(
            level=i,
            coarse_model=coarse_model,
            fine_model=fine_model,
            coarse_proposal=coarse_proposal,
            fine_proposal=fine_proposal,
            kernel=kernel,
            n_samples=n_samples,
            initial_position_coarse=coarse_init,
            initial_position_fine=fine_init,
        )
        levels.append(level)

    return levels


# Example usage
if __name__ == "__main__":
    import sys

    # Simple example with dummy models
    def create_example_levels():
        from samosa.kernels.synce import SYNCEKernel
        from samosa.proposals.gaussianproposal import GaussianRandomWalk

        dim = 5

        # Level 0: fine model only
        fine_model_0 = lambda x: {"log_posterior": -0.5 * np.sum(x**2)}
        fine_proposal_0 = GaussianRandomWalk(np.zeros((dim, 1)), 0.1 * np.eye(dim))
        kernel_0 = SYNCEKernel(None, fine_model_0)  # Coarse model is None for level 0

        # Level 1: coupled chains
        coarse_model_1 = lambda x: {"log_posterior": -0.5 * np.sum(x**2) + 0.1 * np.random.normal()}
        fine_model_1 = lambda x: {"log_posterior": -0.5 * np.sum(x**2)}
        coarse_proposal_1 = GaussianRandomWalk(np.zeros((dim, 1)), 0.1 * np.eye(dim))
        fine_proposal_1 = GaussianRandomWalk(np.zeros((dim, 1)), 0.1 * np.eye(dim))
        kernel_1 = SYNCEKernel(coarse_model_1, fine_model_1)

        levels = [
            MLMCLevel(
                level=0,
                coarse_model=None,
                fine_model=fine_model_0,
                coarse_proposal=None,
                fine_proposal=fine_proposal_0,
                kernel=kernel_0,
                n_samples=1000,
                initial_position_coarse=None,
                initial_position_fine=np.zeros((dim, 1)),
            ),
            MLMCLevel(
                level=1,
                coarse_model=coarse_model_1,
                fine_model=fine_model_1,
                coarse_proposal=coarse_proposal_1,
                fine_proposal=fine_proposal_1,
                kernel=kernel_1,
                n_samples=500,
                initial_position_coarse=np.zeros((dim, 1)),
                initial_position_fine=np.zeros((dim, 1)),
            ),
        ]
        return levels

    # Run example
    if len(sys.argv) > 1 and sys.argv[1] == "example":
        levels = create_example_levels()

        mlmc = SimpleMLMCWrapper(levels=levels, output_dir="./simple_mlmc_example")

        results = mlmc.run()

        if results is not None:  # Only on rank 0
            print(f"MLMC expectation: {results.mlmc_expectation}")
            print(f"MLMC variance: {results.mlmc_variance}")
