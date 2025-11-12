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
import matplotlib.pyplot as plt
from mpi4py import MPI

from samosa.core.kernel import KernelProtocol
from samosa.core.model import ModelProtocol
from samosa.core.proposal import ProposalProtocol
from samosa.core.state import ChainState
from samosa.samplers.coupled_chain import coupledMCMCsampler
from samosa.samplers.single_chain import MCMCsampler
from samosa.utils.post_processing import load_samples, load_coupled_samples, get_position_from_states, scatter_matrix, plot_trace, plot_lag, joint_plots
from samosa.utils.tools import batched_variance

def distribute_level_list_across_ranks(levels: List[int], rank: int, size: int, print_progress: bool = False) -> List[int]:
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
    
    def __init__(self, levels: List[MLMCLevel], output_dir: str = "./mlmc_output", print_progress: bool = True):

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
                print(f"MLMC Sampler Setup: {len(self.levels)} levels, {self.size} MPI processes")
        self.comm.Barrier()
    
    def run(self) -> None:
        """Run only the sampling part - saves samples to files"""

        if self.rank == 0 and self.print_progress:
            print("Starting MLMC sampling...")
            print(f"Total samples: {[level.n_samples for level in self.levels]}")
        
        # Distribute levels across processes
        levels_list = list(range(len(self.levels)))
        my_levels = distribute_level_list_across_ranks(levels_list, self.rank, self.size, self.print_progress)
        
        # Sample assigned levels
        for level_idx in my_levels:
            level = self.levels[level_idx]
            
            if self.print_progress:
                print(f"Process {self.rank}: Starting level {level.level}")
            
            start_time = time.time()
            self._sample_level(level)
            end_time = time.time()
            
            if self.print_progress:
                print(f"Process {self.rank}: Completed level {level.level} in {end_time - start_time:.2f}s")
        
        if self.rank == 0 and self.print_progress:
            print("MLMC sampling complete! Files saved to individual level directories.")
    
    def _sample_level(self, level: MLMCLevel) -> None:
        """Sample a single MLMC level and save to files"""

        level_output_dir = f"{self.output_dir}/level_{level.level}"
        
        if level.level == 0:
            # Level 0: single chain sampling of finest model only
            if self.rank == 0 and self.print_progress:
                print(f"Sampling Level 0: {level.n_samples} samples (fine model only)")
            
            sampler = MCMCsampler(
                kernel=level.kernel,
                proposal=level.fine_proposal,
                initial_position=level.initial_position_fine,
                n_iterations=level.n_samples,
                print_iteration=max(int(level.n_samples // 10), 1),
            )
            
            ar = sampler.run(level_output_dir)
            with open(f"{level_output_dir}/acceptance_rate.txt", "w") as f:
                f.write(f"acceptance_rate: {ar}\n")
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
                print_iteration=max(int(level.n_samples // 10), 1),
            )
            
            ar_coarse, ar_fine = sampler.run(level_output_dir)
            with open(f"{level_output_dir}/acceptance_rates.txt", "w") as f:
                f.write(f"ar_coarse: {ar_coarse}\n")
                f.write(f"ar_fine: {ar_fine}\n")
    
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
    
    def compute_mlmc_estimator(self, burnin_fraction: float = 0.25) -> MLMCResults:
        """
        Compute MLMC estimator from saved samples
        
        Args:
            burnin_fraction: Fraction of samples to discard as burnin (0.0 to 1.0)
        
        Returns:
            MLMCResults object
        """

        if self.rank == 0 and self.print_progress:
            print(f"Computing MLMC estimator with burnin fraction: {burnin_fraction}")
        
        # Distribute levels across ranks for processing
        levels_list = list(range(self.num_levels))
        my_levels = distribute_level_list_across_ranks(levels_list, self.rank, self.size, self.print_progress)
        
        # Process assigned levels
        local_results = {}
        
        for level in my_levels:
            if self.print_progress:
                print(f"Process {self.rank}: Processing level {level}")
                        
            level_dir = f"{self.output_dir}/level_{level}"

            if not os.path.exists(level_dir):
                raise FileNotFoundError(f"Level directory not found: {level_dir}")
                        
            if level == 0:
                samples_fine = load_samples(level_dir)
                samples_coarse = [None] * len(samples_fine)
                
            else:
                samples_coarse, samples_fine = load_coupled_samples(level_dir)
            
            mean_diff, variance_diff, correlations = self._compute_qoi_differences(samples_coarse, samples_fine, burnin_fraction)
            
            # Store results for this level
            local_results[level] = {
                'level': level,
                'mean': mean_diff,
                'variance': variance_diff,
                'correlations': correlations,
                'n_samples': len(samples_fine),
            }
            
            if self.print_progress:
                print(f"Process {self.rank}: Completed level {level}")

        # Gather results from all processes
        all_results = self.comm.gather(local_results, root=0)
        
        # Combine and return results on root process
        if self.rank == 0:
            # Combine results from all processes
            combined_results = {}
            for proc_results in all_results:
                combined_results.update(proc_results)
            
            # Sort by level and extract results
            sorted_levels = sorted(combined_results.keys())
            level_means = [combined_results[i]['mean'] for i in sorted_levels]
            level_variances = [combined_results[i]['variance'] for i in sorted_levels]
            level_correlations = [combined_results[i]['correlations'] for i in sorted_levels]
            samples_per_level = [combined_results[i]['n_samples'] for i in sorted_levels]
            levels_completed = [combined_results[i]['level'] for i in sorted_levels]
            
            # Compute MLMC estimator
            mlmc_expectation = sum(level_means)
            mlmc_variance = sum(var / n_samples for var, n_samples in zip(level_variances, samples_per_level))
            
            # Create results object
            results = MLMCResults(
                level_means=level_means,
                level_variances=level_variances,
                level_correlations=level_correlations,
                mlmc_expectation=mlmc_expectation,
                mlmc_variance=mlmc_variance,
                samples_per_level=samples_per_level,
                levels_completed=levels_completed,
                burnin_fraction=burnin_fraction
            )
            
            # Save results
            with open(f"{self.output_dir}/mlmc_results.pkl", "wb") as f:
                pickle.dump(results, f)
            
            if self.print_progress:
                print("\n" + "="*50)
                print("MLMC ESTIMATION COMPLETE")
                print("="*50)
                print(f"Levels completed: {levels_completed}")
                print(f"Samples per level: {samples_per_level}")
                print(f"Burnin fraction: {burnin_fraction}")
                print(f"MLMC expectation shape: {mlmc_expectation.shape}")
                print(f"MLMC variance shape: {mlmc_variance.shape}")
                print(f"Results saved to: {self.output_dir}/mlmc_results.pkl")
                print("="*50)
        
        return None
    
    def _compute_qoi_differences(self, samples_coarse: List[ChainState], samples_fine: List[ChainState], burnin: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute QoI differences, statistics, and per-dimension correlation for a level"""

        n_samples = len(samples_fine)
        start = int(burnin * n_samples)
        
        differences = []
        qoi_fine_list = []
        qoi_coarse_list = []
        for i in range(start, n_samples):
            qoi_fine = np.atleast_1d(samples_fine[i].qoi)
            qoi_fine_list.append(qoi_fine)
            
            if samples_coarse[i] is not None:
                qoi_coarse = np.atleast_1d(samples_coarse[i].qoi)
                qoi_coarse_list.append(qoi_coarse)
                diff = qoi_fine - qoi_coarse
            else: 
                qoi_coarse_list.append(np.full_like(qoi_fine, np.nan))
                diff = qoi_fine
            
            differences.append(diff)

        differences = np.array(differences).T
        mean_diff = np.mean(differences, axis=1)
        variance_diff = batched_variance(differences)

        # Stack as (n_samples, dim)
        qoi_fine_arr = np.vstack(qoi_fine_list)
        qoi_coarse_arr = np.vstack(qoi_coarse_list)
        valid_mask = ~np.isnan(qoi_coarse_arr).any(axis=1)
        qoi_fine_valid = qoi_fine_arr[valid_mask]
        qoi_coarse_valid = qoi_coarse_arr[valid_mask]

        # Compute per-dimension correlation
        if qoi_fine_valid.shape[0] > 1:
            correlations = np.array([
                np.corrcoef(qoi_fine_valid[:, d], qoi_coarse_valid[:, d])[0, 1]
                for d in range(qoi_fine_valid.shape[1])
            ])
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
            
            print(f"\n{'='*60}")
            print(f"MLMC ESTIMATION SUMMARY")
            print(f"{'='*60}")
            
            print(f"Levels completed: {results.levels_completed}")
            print(f"Samples per level: {results.samples_per_level}")
            print(f"MLMC expectation: {results.mlmc_expectation}")
            print(f"MLMC variance: {results.mlmc_variance}")
            
            print(f"\nLevel-wise statistics:")
            for i, level in enumerate(results.levels_completed):
                print(f"  Level {level}:")
                print(f"    Mean: {results.level_means[i]}")
                print(f"    Variance: {results.level_variances[i]}")
                print(f"    Samples: {results.samples_per_level[i]}")
                print(f"    Correlations: {results.level_correlations[i]}")
            
            print(f"{'='*60}\n")

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
    
    def process_levels(self, levels: List[int], burnin_fraction: float = 0.25, maxlag: int = 500, img_kwargs: Optional[dict] = None):
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
            print(f"\n{'='*70}")
            print(f"MLMC POST-PROCESSING")
            print(f"Processing levels: {levels}")
            print(f"Burnin fraction: {burnin_fraction}")
            print(f"MPI processes: {self.size}")
            print(f"{'='*70}")
        
        # Set default image parameters
        if img_kwargs is None:
            img_kwargs = {
                'label_fontsize': 18,
                'title_fontsize': 20,
                'tick_fontsize': 16,
                'legend_fontsize': 16,
                'img_format': 'png'
            }
        
        # Distribute levels across MPI processes
        my_levels = distribute_level_list_across_ranks(levels, self.rank, self.size, self.print_progress)
        
        if self.print_progress:
            print(f"Process {self.rank}: Assigned levels {my_levels}")
        
        for level in my_levels:

            if self.print_progress:
                print(f"Process {self.rank}: Processing level {level}...")
            
            try:
                # Load samples for this level
                samples_coarse, samples_fine = self._load_samples_for_level(level)
                
                # 1. Level-wise scatter plots
                self._create_level_scatter_plots(level, samples_coarse, samples_fine, img_kwargs, burnin_fraction)
                
                # 2. Level-wise trace plots per dimension  
                self._create_level_trace_plots(level, samples_coarse, samples_fine, img_kwargs, burnin_fraction)
                
                # 3. Level-wise autocorrelation plots per dimension
                self._create_level_autocorr_plots(level, samples_coarse, samples_fine, maxlag, img_kwargs, burnin_fraction)
                
                # 4. Level-wise joint scatter plots
                self._create_level_joint_plots(level, samples_coarse, samples_fine, img_kwargs, burnin_fraction)
                
                if self.print_progress:
                    print(f"Process {self.rank}: Level {level} completed")
                
            except Exception as e:
                if self.print_progress:
                    print(f"Process {self.rank}: Error processing level {level}: {str(e)}")
                continue
                        
        if self.print_progress:
            print(f"\n{'='*70}")
            print(f"POST-PROCESSING COMPLETE")
            print(f"{'='*70}\n")

    def _load_samples_for_level(self, level: int) -> Tuple[List[ChainState], List[ChainState]]:
        """Load samples for a specific level with burnin"""

        # Try to find samples from any rank
        samples_coarse = None
        samples_fine = None
        
        level_dir = f"{self.output_dir}/level_{level}"
        
        if level == 0:
            samples_fine = load_samples(level_dir)
            samples_coarse = [None] * len(samples_fine)

        else:
            samples_coarse, samples_fine = load_coupled_samples(level_dir)
        
        return samples_coarse, samples_fine
    
    def _create_level_scatter_plots(self, level: int, samples_coarse: List[ChainState], samples_fine: List[ChainState], img_kwargs: dict, burnin: float = 0.0):
        """Create scatter plots for a level"""

        level_dir = f"{self.output_dir}/level_{level}"
        
        # Extract positions
        positions_fine = get_position_from_states(samples_fine, burnin=burnin)
        
        if level == 0:
            # Single scatter matrix for level 0
            fig_scatter, _, _ = scatter_matrix(
                samples=[positions_fine],
                sample_labels=[f'Level {level}'],
                img_kwargs=img_kwargs
            )
            
            fig_scatter.savefig(f"{level_dir}/scatter.png", dpi=300, bbox_inches='tight')
            plt.close(fig_scatter)
            
        else:
            # Comparison scatter matrix for level > 0
            positions_coarse = get_position_from_states(samples_coarse, burnin=burnin)
            
            # Combined scatter matrix
            fig_scatter, _, _ = scatter_matrix(
                samples=[positions_coarse, positions_fine],
                sample_labels=[f'Level {level} Coarse', f'Level {level} Fine'],
                img_kwargs=img_kwargs
            )
            
            fig_scatter.savefig(f"{level_dir}/scatter.png", dpi=300, bbox_inches='tight')
            plt.close(fig_scatter)
    
    def _create_level_trace_plots(self, level: int, samples_coarse: List[ChainState], samples_fine: List[ChainState], img_kwargs: dict, burnin: float = 0.0):
        """Create trace plots per dimension for a level"""
        
        level_dir = f"{self.output_dir}/level_{level}"

        # Extract positions
        positions_fine = get_position_from_states(samples_fine, burnin=burnin)

        if level == 0:
            fig, _ = plot_trace([positions_fine], img_kwargs=img_kwargs)
            fig.savefig(f"{level_dir}/trace.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
                    
        if level > 0:
            positions_coarse = get_position_from_states(samples_coarse, burnin=burnin)
            fig,_ = plot_trace([positions_coarse, positions_fine], img_kwargs=img_kwargs)
            fig.savefig(f"{level_dir}/trace.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    def _create_level_autocorr_plots(self, level: int, samples_coarse: List[ChainState], samples_fine: List[ChainState], maxlag: int, img_kwargs: dict, burnin: float = 0.0):
        """Create autocorrelation plots per dimension for a level"""

        level_dir = f"{self.output_dir}/level_{level}"
        
        # Extract positions
        positions_fine = get_position_from_states(samples_fine, burnin=burnin)

        if level == 0:
            fig, _, _, _ = plot_lag([positions_fine], maxlag=maxlag, img_kwargs=img_kwargs)
            fig.savefig(f"{level_dir}/autocorr.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        if level > 0:
            positions_coarse = get_position_from_states(samples_coarse, burnin=burnin)
            fig, _, _, _ = plot_lag([positions_coarse, positions_fine], maxlag=maxlag, img_kwargs=img_kwargs)
            fig.savefig(f"{level_dir}/autocorr.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    def _create_level_joint_plots(self, level: int, samples_coarse: List[ChainState], samples_fine: List[ChainState], img_kwargs: dict, burnin: float = 0.0):
        """Create joint scatter plots for a level (only for level > 0)"""

        level_dir = f"{self.output_dir}/level_{level}"
        
        if level == 0:
            return  # No joint plots for level 0
        
        # Extract positions
        positions_coarse = get_position_from_states(samples_coarse, burnin=burnin)
        positions_fine = get_position_from_states(samples_fine, burnin=burnin)
        
        # Create joint plots
        fig_joints = joint_plots(
            samples=[positions_coarse, positions_fine],
            img_kwargs=img_kwargs
        )
        
        # Save joint plots
        for i, fig in enumerate(fig_joints):
            fig.savefig(f"{level_dir}/joint_dim_{i+1}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

# Utility function for easy setup (unchanged)
def create_mlmc_levels(
    models: List[Tuple[ModelProtocol, ModelProtocol]],
    proposals: List[Tuple[ProposalProtocol, ProposalProtocol]],
    kernels: List[KernelProtocol],
    samples_per_level: List[int],
    initial_positions: List[Tuple[np.ndarray, np.ndarray]],
) -> List[MLMCLevel]:
    """Utility function to create MLMC levels from lists of components."""
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

    from samosa.utils.tools import lognormpdf

    def gaussian_model(x: np.ndarray, level: int) -> dict[str, any]:
        """
        Banana model function
        """
        output = {}
        # Just use the log_banana function to compute the log posterior
        log_posterior = lognormpdf(x, mean=np.array([[2**(-level+2)],[3**(-level+2)]]), cov=np.array([[2, 2**(-level)], [2**(-level), 1]]))

        output['log_posterior'] = log_posterior

        # If you want to compute the qoi, cost_model_output etc. you can do it like this
        cost = 1
        qoi = x[0]

        output['cost'] = cost
        output['qoi'] = qoi

        return output

    from samosa.proposals.gaussianproposal import GaussianRandomWalk
    from samosa.proposals.adapters import GlobalAdapter
    from samosa.core.proposal import AdaptiveProposal
    from samosa.kernels.synce import SYNCEKernel
    from samosa.kernels.metropolis import MetropolisHastingsKernel

    L = 3
    models_list = [lambda x, level=i: gaussian_model(x, level) for i in range(L)]
    models = [(None, models_list[0])] + [(models_list[i], models_list[i + 1]) for i in range(L - 1)]
    proposal = GaussianRandomWalk(mu=np.zeros((2,1)), sigma=3 * np.eye(2))
    adapter = GlobalAdapter(ar = 0.44, adapt_end=10000)
    adaptive_proposal = AdaptiveProposal(base_proposal=proposal, adapter=adapter)
    proposals = [(None, adaptive_proposal)] + [(adaptive_proposal, adaptive_proposal) for _ in range(L - 1)]
    kernels = [SYNCEKernel(coarse_model=coarse, fine_model=fine) if coarse is not None else MetropolisHastingsKernel(model=fine) for coarse, fine in models]
    samples_per_level = [500000, 50000, 10000]
    initial_positions = [(None, np.array([[0.0],[0.0]]))] + [(np.array([[0.0],[0.0]]), np.array([[0.0],[0.0]])) for _ in range(L - 1)]

    levels = create_mlmc_levels(
        models=models,
        proposals=proposals,
        kernels=kernels,
        samples_per_level=samples_per_level,
        initial_positions=initial_positions,
    )
    
    mlmc = MLMCSampler(levels=levels, output_dir='mlmc_gaussian_example', print_progress=True)
    mlmc.run()

    mlmc_calc = MLMCCalculator(output_dir='mlmc_gaussian_example', num_levels=L, print_progress=True)
    mlmc_calc.compute_mlmc_estimator(burnin_fraction=0.3)
    mlmc_calc.print_mlmc_summary()

    # mlmc_post = MLMCPostProcessor(output_dir='mlmc_gaussian_example', print_progress=True)
    # mlmc_post.process_levels(levels=list(range(L)), burnin_fraction=0.3)
