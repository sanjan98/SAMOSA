"""
Simple Multi-Level Monte Carlo wrapper.

This wrapper manages multiple fidelity levels and distributes them across MPI
processes. Each coupled chain (level) runs on a separate MPI rank when enough
ranks are available.

Nested parallelism (models using MPI internally)
-----------------------------------------------
SAMOSA does not run or manage MPI inside the model. If your model (e.g. a PDE
solver) uses MPI internally, you must implement one of these patterns:

Option A – Sub-communicator (in-process MPI model)
  Use ``split_comm_for_levels(MPI.COMM_WORLD, num_levels, rank)`` to get a
  sub-communicator for the current level. You are responsible for passing that
  sub_comm into your model (e.g. store it in the model at construction, or use a
  closure over the rank's sub_comm). When the kernel calls the model, the model
  uses that sub_comm for its internal solver (e.g. domain decomposition). SAMOSA
  does not pass a communicator into the model; the model protocol is unchanged.

  Example layout with 8 ranks and 4 levels:
    - Ranks 0-1: level 0 (sub_comm size 2)
    - Ranks 2-3: level 1 (sub_comm size 2)
    - etc.
  Coarse and fine within a level run sequentially, so they share the same
  sub_comm.

Option B – Subprocess (MPI executable model)
  If the model is an external MPI executable (e.g. ``mpirun -n 4 solver.exe``),
  run it in a subprocess (e.g. ``subprocess.run(...)``) and read outputs.
  SAMOSA's MPI is then used only to distribute levels across ranks; there is no
  shared communicator with the solver in the same process. This also works when
  the MPI executable needs a different Python environment or packages.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from samosa.core.kernel import CoupledKernelBase
from samosa.core.state import ChainState
from samosa.kernels.delayedrejection import DelayedRejectionKernel
from samosa.kernels.metropolis import MetropolisHastingsKernel
from samosa.samplers.coupled_chain import CoupledChainSampler
from samosa.samplers.single_chain import SingleChainSampler
from samosa.utils.post_processing import (
    load_samples,
    load_coupled_samples,
    get_position_from_states,
    scatter_matrix,
    plot_trace,
    plot_lag,
    joint_plots,
)
from samosa.utils.tools import _batched_variance

logger = logging.getLogger(__name__)

# Type alias for kernels: level 0 uses single-chain, level > 0 uses coupled
SingleChainKernel = Union[MetropolisHastingsKernel, DelayedRejectionKernel]
LevelKernel = Union[SingleChainKernel, CoupledKernelBase]


def distribute_level_list_across_ranks(
    levels: List[int], rank: int, size: int, print_progress: bool = False
) -> List[int]:
    """
    Distribute a specific list of levels across MPI processes.

    Args:
        levels: List of specific level indices to distribute
        rank: Current MPI rank
        size: Total number of MPI processes
        print_progress: Whether to print distribution info (only on rank 0)

    Returns:
        List of level indices assigned to this rank from the input list
    """
    num_levels = len(levels)
    levels_per_process = num_levels // size
    remainder = num_levels % size

    # Give extra levels to first 'remainder' processes
    if rank < remainder:
        start_idx = rank * (levels_per_process + 1)
        n_levels = levels_per_process + 1
    else:
        start_idx = rank * levels_per_process + remainder
        n_levels = levels_per_process

    my_level_indices = list(range(start_idx, start_idx + n_levels))
    my_levels = [levels[i] for i in my_level_indices if i < len(levels)]

    if rank == 0 and print_progress:
        print("Level distribution for post-processing across MPI processes:")
        for r in range(size):
            if r < remainder:
                r_start = r * (levels_per_process + 1)
                r_n = levels_per_process + 1
            else:
                r_start = r * levels_per_process + remainder
                r_n = levels_per_process
            r_level_indices = list(range(r_start, r_start + r_n))
            r_levels = [levels[i] for i in r_level_indices if i < len(levels)]
            print(f"  Process {r}: levels {r_levels}")

    return my_levels


def split_comm_for_levels(
    comm: MPI.Comm,
    num_levels: int,
    rank: int,
) -> Tuple[MPI.Comm, int]:
    """
    Split MPI communicator so each level gets a sub-communicator for nested parallelism.

    Use this when model evaluations (e.g., CFD solvers) use MPI internally. Each level
    runs on a disjoint subset of ranks; those ranks form a sub-communicator that the
    model can use for its internal parallelism.

    Example: 8 ranks, 4 levels -> ranks 0-1 for level 0, 2-3 for level 1, etc.
    Each level's model receives its sub-comm (size 2) for CFD parallelism.

    Args:
        comm: MPI communicator (e.g., MPI.COMM_WORLD).
        num_levels: Number of MLMC levels.
        rank: Current rank in comm.

    Returns:
        Tuple of (sub_comm, sub_rank). Ranks not assigned to any level get
        sub_comm=MPI.COMM_NULL. Assigned ranks get a sub-communicator and their
        rank within it.
    """
    size = comm.Get_size()
    if size < num_levels:
        # Not enough ranks for splitting; each level uses full comm or single rank
        return comm, rank
    ranks_per_level = size // num_levels
    level_color = rank // ranks_per_level
    if level_color >= num_levels:
        return MPI.COMM_NULL, -1
    sub_comm = comm.Split(level_color, rank)
    sub_rank = sub_comm.Get_rank()
    return sub_comm, sub_rank


@dataclass
class MLMCLevel:
    """Configuration for a single MLMC level.

    Level 0 uses a single-chain kernel (MetropolisHastingsKernel or
    DelayedRejectionKernel); level > 0 uses a coupled kernel (CoupledKernelBase).
    The kernel already encapsulates model(s) and proposal(s).
    """

    level: int
    kernel: LevelKernel
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
    level_correlations: List[np.ndarray]  # Correlations for each level
    mlmc_expectation: np.ndarray  # Final MLMC estimator
    mlmc_variance: np.ndarray  # MLMC estimator variance
    samples_per_level: List[int]
    levels_completed: List[int]
    burnin_fraction: float = 0.0


class MLMCSampler:
    """
    MLMC Sampler - handles only the sampling part
    """

    def __init__(
        self,
        levels: List[MLMCLevel],
        output_dir: str = "./mlmc_output",
        print_progress: bool = True,
        print_iteration: Optional[int] = None,
        save_iteration: Optional[int] = None,
    ):

        self.levels = sorted(levels, key=lambda x: x.level)
        self.output_dir = output_dir
        self.print_progress = print_progress
        self.print_iteration = print_iteration
        self.save_iteration = save_iteration
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
                logger.info(
                    "MLMC Sampler Setup: %s levels, %s MPI processes",
                    len(self.levels),
                    self.size,
                )
        self.comm.Barrier()

    def run(self) -> None:
        """Run only the sampling part - saves samples to files"""

        if self.rank == 0 and self.print_progress:
            logger.info("Starting MLMC sampling...")
            logger.info("Total samples: %s", [level.n_samples for level in self.levels])

        # Distribute levels across processes
        levels_list = list(range(len(self.levels)))
        my_levels = distribute_level_list_across_ranks(
            levels_list, self.rank, self.size, self.print_progress
        )

        # Sample assigned levels
        for level_idx in my_levels:
            level = self.levels[level_idx]

            if self.print_progress:
                logger.info("Process %s: Starting level %s", self.rank, level.level)

            start_time = time.time()
            self._sample_level(level)
            end_time = time.time()

            if self.print_progress:
                logger.info(
                    "Process %s: Completed level %s in %.2fs",
                    self.rank,
                    level.level,
                    end_time - start_time,
                )

        if self.rank == 0 and self.print_progress:
            logger.info(
                "MLMC sampling complete! Files saved to individual level directories."
            )

    def _sample_level(self, level: MLMCLevel) -> None:
        """Sample a single MLMC level and save to files."""
        print_iter = self.print_iteration
        save_iter = self.save_iteration
        level_output_dir = f"{self.output_dir}/level_{level.level}"
        if level.level == 0:
            # Level 0: single chain sampling of finest model only
            if level.initial_position_fine is None:
                raise ValueError("Level 0 requires initial_position_fine")
            if self.rank == 0 and self.print_progress:
                logger.info(
                    "Sampling Level 0: %s samples (fine model only)", level.n_samples
                )

            sampler = SingleChainSampler(
                kernel=cast(SingleChainKernel, level.kernel),
                initial_position=level.initial_position_fine,
                n_iterations=level.n_samples,
                print_iteration=print_iter,
                save_iteration=save_iter,
                restart=level.restart_fine,
            )

            ar = sampler.run(level_output_dir)
            with open(f"{level_output_dir}/acceptance_rate.txt", "w") as f:
                f.write(f"acceptance_rate: {ar if ar is not None else 0.0}\n")
        else:
            # Level > 0: coupled chain sampling
            if (
                level.initial_position_coarse is None
                or level.initial_position_fine is None
            ):
                raise ValueError(
                    "Level > 0 requires both initial_position_coarse and initial_position_fine"
                )
            if self.rank == 0 and self.print_progress:
                logger.info(
                    "Sampling Level %s: %s samples (coupled chains)",
                    level.level,
                    level.n_samples,
                )

            sampler = CoupledChainSampler(
                kernel=cast(CoupledKernelBase, level.kernel),
                initial_position_coarse=level.initial_position_coarse,
                initial_position_fine=level.initial_position_fine,
                n_iterations=level.n_samples,
                print_iteration=print_iter,
                save_iteration=save_iter,
                restart_coarse=level.restart_coarse,
                restart_fine=level.restart_fine,
            )

            result = sampler.run(level_output_dir)
            ar_coarse = result[0] if result is not None else 0.0
            ar_fine = result[1] if result is not None else 0.0
            with open(f"{level_output_dir}/acceptance_rates.txt", "w") as f:
                f.write(f"ar_coarse: {ar_coarse}\n")
                f.write(f"ar_fine: {ar_fine}\n")

    def _validate_levels(self):
        """Validate the MLMC level configuration."""
        single_chain_types = (MetropolisHastingsKernel, DelayedRejectionKernel)
        for level in self.levels:
            if level.level == 0:
                if not isinstance(level.kernel, single_chain_types):
                    raise ValueError(
                        "Level 0 must use a single-chain kernel "
                        "(MetropolisHastingsKernel or DelayedRejectionKernel)"
                    )
                if level.initial_position_fine is None:
                    raise ValueError("Level 0 requires initial_position_fine")
            else:
                if not isinstance(level.kernel, CoupledKernelBase):
                    raise ValueError(
                        f"Level {level.level} must use a CoupledKernelBase kernel"
                    )
                if (
                    level.initial_position_coarse is None
                    or level.initial_position_fine is None
                ):
                    raise ValueError(
                        f"Level {level.level} requires both initial_position_coarse "
                        "and initial_position_fine"
                    )

        # Check level numbering
        expected_levels = list(range(len(self.levels)))
        actual_levels = [level.level for level in self.levels]
        if actual_levels != expected_levels:
            raise ValueError(
                f"Levels must be numbered 0, 1, 2, ... Got {actual_levels}"
            )


class MLMCCalculator:
    """
    MLMC Calculator - loads saved samples and computes MLMC estimator
    """

    def __init__(self, output_dir: str, num_levels: int, print_progress: bool = True):
        self.output_dir = output_dir
        self.num_levels = num_levels
        self.print_progress = print_progress

        # MPI setup (for loading distributed files)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def compute_mlmc_estimator(
        self, burnin_fraction: float = 0.25
    ) -> Optional[MLMCResults]:
        """
        Compute MLMC estimator from saved samples

        Args:
            burnin_fraction: Fraction of samples to discard as burnin (0.0 to 1.0)

        Returns:
            MLMCResults object
        """

        if self.rank == 0 and self.print_progress:
            logger.info(
                "Computing MLMC estimator with burnin fraction: %s", burnin_fraction
            )

        # Distribute levels across ranks for processing
        levels_list = list(range(self.num_levels))
        my_levels = distribute_level_list_across_ranks(
            levels_list, self.rank, self.size, self.print_progress
        )

        # Process assigned levels
        local_results = {}

        for level in my_levels:
            if self.print_progress:
                logger.info("Process %s: Processing level %s", self.rank, level)

            level_dir = f"{self.output_dir}/level_{level}"

            if not os.path.exists(level_dir):
                raise FileNotFoundError(f"Level directory not found: {level_dir}")

            if level == 0:
                samples_fine = load_samples(level_dir)
                samples_coarse = [None] * len(samples_fine)

            else:
                samples_coarse, samples_fine = load_coupled_samples(level_dir)

            mean_diff, variance_diff, correlations = self._compute_qoi_differences(
                samples_coarse, samples_fine, burnin_fraction
            )

            # Store results for this level
            local_results[level] = {
                "level": level,
                "mean": mean_diff,
                "variance": variance_diff,
                "correlations": correlations,
                "n_samples": len(samples_fine),
            }

            if self.print_progress:
                logger.info("Process %s: Completed level %s", self.rank, level)

        # Gather results from all processes
        all_results: Optional[List[dict]] = self.comm.gather(local_results, root=0)

        # Combine and return results on root process
        if self.rank == 0 and all_results is not None:
            # Combine results from all processes
            combined_results = {}
            for proc_results in all_results:
                combined_results.update(proc_results)

            # Sort by level and extract results
            sorted_levels = sorted(combined_results.keys())
            level_means = [combined_results[i]["mean"] for i in sorted_levels]
            level_variances = [combined_results[i]["variance"] for i in sorted_levels]
            level_correlations = [
                combined_results[i]["correlations"] for i in sorted_levels
            ]
            samples_per_level = [
                combined_results[i]["n_samples"] for i in sorted_levels
            ]
            levels_completed = [combined_results[i]["level"] for i in sorted_levels]

            # Compute MLMC estimator (ensure ndarray for scalar QoI)
            mlmc_expectation = np.atleast_1d(np.asarray(sum(level_means)))
            mlmc_variance = np.atleast_1d(
                np.asarray(
                    sum(
                        var / n_samples
                        for var, n_samples in zip(level_variances, samples_per_level)
                    )
                )
            )

            # Create results object
            results = MLMCResults(
                level_means=level_means,
                level_variances=level_variances,
                level_correlations=level_correlations,
                mlmc_expectation=mlmc_expectation,
                mlmc_variance=mlmc_variance,
                samples_per_level=samples_per_level,
                levels_completed=levels_completed,
                burnin_fraction=burnin_fraction,
            )

            # Save results
            with open(f"{self.output_dir}/mlmc_results.pkl", "wb") as f:
                pickle.dump(results, f)

            if self.print_progress:
                logger.info("=" * 50)
                logger.info("MLMC ESTIMATION COMPLETE")
                logger.info("Levels completed: %s", levels_completed)
                logger.info("Samples per level: %s", samples_per_level)
                logger.info("Burnin fraction: %s", burnin_fraction)
                logger.info("MLMC expectation shape: %s", mlmc_expectation.shape)
                logger.info("MLMC variance shape: %s", mlmc_variance.shape)
                logger.info("Results saved to: %s/mlmc_results.pkl", self.output_dir)

            return results
        return None

    def _compute_qoi_differences(
        self,
        samples_coarse: Sequence[Optional[ChainState]],
        samples_fine: List[ChainState],
        burnin: float = 0.25,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute QoI differences, statistics, and per-dimension correlation for a level"""

        n_samples = len(samples_fine)
        start = int(burnin * n_samples)

        differences = []
        qoi_fine_list = []
        qoi_coarse_list = []
        for i in range(start, n_samples):
            qoi_f = samples_fine[i].qoi
            qoi_fine = np.atleast_1d(
                qoi_f if qoi_f is not None else samples_fine[i].position
            ).flatten()
            qoi_fine_list.append(qoi_fine)

            coarse_i = samples_coarse[i]
            if coarse_i is not None:
                qoi_c = coarse_i.qoi
                qoi_coarse = np.atleast_1d(
                    qoi_c if qoi_c is not None else coarse_i.position
                ).flatten()
                qoi_coarse_list.append(qoi_coarse)
                diff = qoi_fine - qoi_coarse
            else:
                qoi_coarse_list.append(np.full_like(qoi_fine, np.nan))
                diff = qoi_fine

            differences.append(diff)

        # Stack as (d, N) for batched_variance
        differences = np.vstack(differences).T
        mean_diff = np.mean(differences, axis=1)
        variance_diff = _batched_variance(differences)

        # Stack as (n_samples, dim)
        qoi_fine_arr = np.vstack(qoi_fine_list)
        qoi_coarse_arr = np.vstack(qoi_coarse_list)
        valid_mask = ~np.isnan(qoi_coarse_arr).any(axis=1)
        qoi_fine_valid = qoi_fine_arr[valid_mask]
        qoi_coarse_valid = qoi_coarse_arr[valid_mask]

        # Compute per-dimension correlation (suppress divide-by-zero for constant QoI)
        if qoi_fine_valid.shape[0] > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                correlations = np.array(
                    [
                        np.corrcoef(qoi_fine_valid[:, d], qoi_coarse_valid[:, d])[0, 1]
                        for d in range(qoi_fine_valid.shape[1])
                    ]
                )
            correlations = np.nan_to_num(correlations, nan=1.0, posinf=1.0, neginf=-1.0)
        else:
            correlations = np.full(qoi_fine_valid.shape[1], 1.0)

        return mean_diff, variance_diff, correlations

    def load_mlmc_results(self) -> MLMCResults:
        """Load MLMC results"""
        results_file = f"{self.output_dir}/mlmc_results.pkl"
        with open(results_file, "rb") as f:
            return pickle.load(f)

    def print_mlmc_summary(self):
        """Print summary of MLMC results"""

        if self.rank == 0:
            results = self.load_mlmc_results()

            logger.info("=" * 60)
            logger.info("MLMC ESTIMATION SUMMARY")
            logger.info("Levels completed: %s", results.levels_completed)
            logger.info("Samples per level: %s", results.samples_per_level)
            logger.info("MLMC expectation: %s", results.mlmc_expectation)
            logger.info("MLMC variance: %s", results.mlmc_variance)
            logger.info("Level-wise statistics:")
            for i, level in enumerate(results.levels_completed):
                logger.info("  Level %s:", level)
                logger.info("    Mean: %s", results.level_means[i])
                logger.info("    Variance: %s", results.level_variances[i])
                logger.info("    Samples: %s", results.samples_per_level[i])
                logger.info("    Correlations: %s", results.level_correlations[i])
            logger.info("=" * 60)


class MLMCPostProcessor:
    """
    MLMC Post-processor - handles visualization and analysis using existing plotting functions
    """

    def __init__(self, output_dir: str, print_progress: bool = True):
        self.output_dir = output_dir
        self.print_progress = print_progress

        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.comm.Barrier()  # Ensure directory is created before other ranks proceed

    def process_levels(
        self,
        levels: List[int],
        burnin_fraction: float = 0.25,
        maxlag: int = 500,
        img_kwargs: Optional[dict] = None,
    ):
        """
        Comprehensive post-processing for specified levels distributed across MPI processes

        Args:
            levels: List of level indices to process
            burnin_fraction: Fraction of samples to discard as burnin
            maxlag: Maximum lag for autocorrelation plots
            img_kwargs: Image formatting parameters
            include_joint_plots: Whether to generate joint scatter plots
        """

        if self.rank == 0 and self.print_progress:
            logger.info("=" * 70)
            logger.info("MLMC POST-PROCESSING")
            logger.info("Processing levels: %s", levels)
            logger.info("Burnin fraction: %s", burnin_fraction)
            logger.info("MPI processes: %s", self.size)
            logger.info("=" * 70)

        # Set default image parameters
        if img_kwargs is None:
            img_kwargs = {
                "label_fontsize": 18,
                "title_fontsize": 20,
                "tick_fontsize": 16,
                "legend_fontsize": 16,
                "img_format": "png",
            }

        # Distribute levels across MPI processes
        my_levels = distribute_level_list_across_ranks(
            levels, self.rank, self.size, self.print_progress
        )

        if self.print_progress:
            logger.info("Process %s: Assigned levels %s", self.rank, my_levels)

        for level in my_levels:
            if self.print_progress:
                logger.info("Process %s: Processing level %s...", self.rank, level)
            level_dir = f"{self.output_dir}/level_{level}"
            self.fig_dir = f"{level_dir}/figures"
            os.makedirs(self.fig_dir, exist_ok=True)
            try:
                # Load samples for this level
                samples_coarse, samples_fine = self._load_samples_for_level(level)

                # 1. Level-wise scatter plots
                self._create_level_scatter_plots(
                    level, samples_coarse, samples_fine, img_kwargs, burnin_fraction
                )

                # 2. Level-wise trace plots per dimension
                self._create_level_trace_plots(
                    level, samples_coarse, samples_fine, img_kwargs, burnin_fraction
                )

                # 3. Level-wise autocorrelation plots per dimension
                self._create_level_autocorr_plots(
                    level,
                    samples_coarse,
                    samples_fine,
                    maxlag,
                    img_kwargs,
                    burnin_fraction,
                )

                # 4. Level-wise joint scatter plots
                self._create_level_joint_plots(
                    level, samples_coarse, samples_fine, img_kwargs, burnin_fraction
                )

                if self.print_progress:
                    logger.info("Process %s: Level %s completed", self.rank, level)

            except Exception as e:
                if self.print_progress:
                    logger.warning(
                        "Process %s: Error processing level %s: %s",
                        self.rank,
                        level,
                        str(e),
                    )
                continue

        if self.print_progress:
            logger.info("=" * 70)
            logger.info("POST-PROCESSING COMPLETE")
            logger.info("=" * 70)

    def _load_samples_for_level(
        self, level: int
    ) -> Tuple[List[Optional[ChainState]], List[ChainState]]:
        """Load samples for a specific level with burnin"""

        level_dir = f"{self.output_dir}/level_{level}"

        if level == 0:
            samples_fine = load_samples(level_dir)
            samples_coarse = [None] * len(samples_fine)
        else:
            samples_coarse, samples_fine = load_coupled_samples(level_dir)

        return (
            cast(List[Optional[ChainState]], samples_coarse),
            samples_fine,
        )

    def _create_level_scatter_plots(
        self,
        level: int,
        samples_coarse: List[Optional[ChainState]],
        samples_fine: List[ChainState],
        img_kwargs: dict,
        burnin: float = 0.0,
    ):
        """Create scatter plots for a level"""
        # Extract positions
        positions_fine = get_position_from_states(samples_fine, burnin=burnin)

        if level == 0:
            # Single scatter matrix for level 0
            fig_scatter, _, _ = scatter_matrix(
                samples=[positions_fine],
                sample_labels=[f"Level {level}"],
                img_kwargs=img_kwargs,
            )

            fig_scatter.savefig(
                f"{self.fig_dir}/scatter.png", dpi=300, bbox_inches="tight"
            )
            plt.close(fig_scatter)

        else:
            # Comparison scatter matrix for level > 0 (samples_coarse is List[ChainState])
            positions_coarse = get_position_from_states(
                cast(List[ChainState], samples_coarse), burnin=burnin
            )

            # Combined scatter matrix
            fig_scatter, _, _ = scatter_matrix(
                samples=[positions_coarse, positions_fine],
                sample_labels=[f"Level {level} Coarse", f"Level {level} Fine"],
                img_kwargs=img_kwargs,
            )

            fig_scatter.savefig(
                f"{self.fig_dir}/scatter.png", dpi=300, bbox_inches="tight"
            )
            plt.close(fig_scatter)

    def _create_level_trace_plots(
        self,
        level: int,
        samples_coarse: List[Optional[ChainState]],
        samples_fine: List[ChainState],
        img_kwargs: dict,
        burnin: float = 0.0,
    ):
        """Create trace plots per dimension for a level"""
        # Extract positions
        positions_fine = get_position_from_states(samples_fine, burnin=burnin)

        if level == 0:
            fig, _ = plot_trace([positions_fine], img_kwargs=img_kwargs)
            fig.savefig(f"{self.fig_dir}/trace.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

        if level > 0:
            positions_coarse = get_position_from_states(
                cast(List[ChainState], samples_coarse), burnin=burnin
            )
            fig, _ = plot_trace(
                [positions_coarse, positions_fine], img_kwargs=img_kwargs
            )
            fig.savefig(f"{self.fig_dir}/trace.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    def _create_level_autocorr_plots(
        self,
        level: int,
        samples_coarse: List[Optional[ChainState]],
        samples_fine: List[ChainState],
        maxlag: int,
        img_kwargs: dict,
        burnin: float = 0.0,
    ):
        """Create autocorrelation plots per dimension for a level"""
        # Extract positions
        positions_fine = get_position_from_states(samples_fine, burnin=burnin)

        if level == 0:
            fig, _, _, _ = plot_lag(
                [positions_fine], maxlag=maxlag, img_kwargs=img_kwargs
            )
            fig.savefig(f"{self.fig_dir}/autocorr.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

        if level > 0:
            positions_coarse = get_position_from_states(
                cast(List[ChainState], samples_coarse), burnin=burnin
            )
            fig, _, _, _ = plot_lag(
                [positions_coarse, positions_fine], maxlag=maxlag, img_kwargs=img_kwargs
            )
            fig.savefig(f"{self.fig_dir}/autocorr.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    def _create_level_joint_plots(
        self,
        level: int,
        samples_coarse: List[Optional[ChainState]],
        samples_fine: List[ChainState],
        img_kwargs: dict,
        burnin: float = 0.0,
    ):
        """Create joint scatter plots for a level (only for level > 0)"""

        if level == 0:
            return  # No joint plots for level 0

        # Extract positions (samples_coarse is List[ChainState] when level > 0)
        positions_coarse = get_position_from_states(
            cast(List[ChainState], samples_coarse), burnin=burnin
        )
        positions_fine = get_position_from_states(samples_fine, burnin=burnin)

        # Create joint plots
        fig_joints = joint_plots(
            samples=[positions_coarse, positions_fine], img_kwargs=img_kwargs
        )

        # Save joint plots
        for i, fig in enumerate(fig_joints):
            fig.savefig(
                f"{self.fig_dir}/joint_dim_{i + 1}.png", dpi=300, bbox_inches="tight"
            )
            plt.close(fig)


def create_mlmc_levels(
    kernels: Sequence[LevelKernel],
    samples_per_level: Sequence[int],
    initial_positions: Sequence[Tuple[Optional[np.ndarray], np.ndarray]],
    restart_coarse: Optional[Sequence[Optional[List[ChainState]]]] = None,
    restart_fine: Optional[Sequence[Optional[List[ChainState]]]] = None,
) -> List[MLMCLevel]:
    """Create MLMC levels from pre-built kernels and per-level config.

    The user supplies one kernel per level (each kernel already has model and
    proposal inside). Level 0 must use a single-chain kernel; level > 0 must
    use a coupled kernel.

    Args:
        kernels: One kernel per level (MetropolisHastingsKernel or
            DelayedRejectionKernel for level 0; CoupledKernelBase for level > 0).
        samples_per_level: Number of samples to draw per level.
        initial_positions: Per level (coarse_init, fine_init). For level 0,
            coarse_init must be None.
        restart_coarse: Optional restart samples per level (None for level 0).
        restart_fine: Optional restart samples per level.

    Returns:
        List of MLMCLevel instances.
    """
    n = len(kernels)
    if not (n == len(samples_per_level) == len(initial_positions)):
        raise ValueError(
            "kernels, samples_per_level, and initial_positions must have the same length"
        )
    if restart_coarse is not None and len(restart_coarse) != n:
        raise ValueError("restart_coarse must have same length as kernels")
    if restart_fine is not None and len(restart_fine) != n:
        raise ValueError("restart_fine must have same length as kernels")

    levels = []
    for i in range(n):
        coarse_init, fine_init = initial_positions[i]
        rc = restart_coarse[i] if restart_coarse is not None else None
        rf = restart_fine[i] if restart_fine is not None else None
        levels.append(
            MLMCLevel(
                level=i,
                kernel=kernels[i],
                n_samples=samples_per_level[i],
                initial_position_coarse=coarse_init,
                initial_position_fine=fine_init,
                restart_coarse=rc,
                restart_fine=rf,
            )
        )
    return levels


# Example usage
if __name__ == "__main__":
    from samosa.utils.tools import lognormpdf

    def gaussian_model(x: np.ndarray, level: int) -> dict:
        """Gaussian model for MLMC levels."""
        output = {}
        log_posterior = lognormpdf(
            x,
            mean=np.array([[2 ** (-level + 2)], [3 ** (-level + 2)]]),
            cov=np.array([[2, 2 ** (-level)], [2 ** (-level), 1]]),
        )
        output["log_posterior"] = log_posterior
        output["cost"] = 2**level
        output["qoi"] = x[0]
        return output

    from samosa.proposals.gaussianproposal import GaussianRandomWalk
    from samosa.proposals.adapters import GlobalAdapter
    from samosa.proposals.coupled_proposals import SynceCoupling
    from samosa.core.proposal import AdaptiveProposal
    from samosa.core.kernel import CoupledKernel
    from samosa.kernels.metropolis import MetropolisHastingsKernel

    L = 3  # max level index: levels 0, 1, ..., L (L+1 levels total)
    proposal = GaussianRandomWalk(mu=np.zeros((2, 1)), cov=3 * np.eye(2))
    adapter = GlobalAdapter(target_ar=0.44, adapt_end=10000)
    adaptive_proposal = AdaptiveProposal(base_proposal=proposal, adapter=adapter)
    kernels = [
        MetropolisHastingsKernel(
            model=lambda params: gaussian_model(params, level=0),
            proposal=adaptive_proposal,
        )
    ] + [
        CoupledKernel(
            coarse_model=lambda params, i=i: gaussian_model(params, level=i - 1),
            fine_model=lambda params, i=i: gaussian_model(params, level=i),
            coupled_proposal=SynceCoupling(adaptive_proposal, adaptive_proposal),
        )
        for i in range(1, L + 1)  # levels 1, 2, ..., L (L levels total)
    ]

    samples_per_level = [50000] * (L + 1)
    initial_positions = [(None, np.random.randn(2, 1))] + [
        (np.random.randn(2, 1), np.random.randn(2, 1)) for _ in range(L)
    ]

    levels = create_mlmc_levels(
        kernels=kernels,
        samples_per_level=samples_per_level,
        initial_positions=initial_positions,
    )

    mlmc = MLMCSampler(
        levels=levels, output_dir="mlmc_gaussian_example", print_progress=True
    )
    mlmc.run()

    mlmc_calc = MLMCCalculator(
        output_dir="mlmc_gaussian_example", num_levels=L + 1, print_progress=True
    )
    mlmc_calc.compute_mlmc_estimator(burnin_fraction=0.3)
    mlmc_calc.print_mlmc_summary()

    mlmc_post = MLMCPostProcessor(
        output_dir="mlmc_gaussian_example", print_progress=True
    )
    mlmc_post.process_levels(levels=list(range(L + 1)), burnin_fraction=0.3)
