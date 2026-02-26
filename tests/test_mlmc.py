"""
Unit tests for the MLMC (Multi-Level Monte Carlo) module.

Tests distribute_level_list_across_ranks, split_comm_for_levels, create_mlmc_levels,
MLMCCalculator._compute_qoi_differences, MLMCSampler validation, and integration
with saved samples. MPI-dependent tests run with COMM_WORLD (typically size 1 when
pytest is run without mpirun).
"""

import os
import pickle
import tempfile
import numpy as np
import pytest
from mpi4py import MPI

from samosa.core.mlmc import (
    MLMCCalculator,
    MLMCLevel,
    MLMCResults,
    MLMCSampler,
    create_mlmc_levels,
    distribute_level_list_across_ranks,
    split_comm_for_levels,
)
from samosa.core.state import ChainState
from samosa.core.model import ModelProtocol
from samosa.core.proposal import Proposal
from samosa.core.kernel import CoupledKernelBase
from samosa.kernels.metropolis import MetropolisHastingsKernel
from samosa.proposals.gaussianproposal import GaussianRandomWalk
from samosa.proposals.adapters import GlobalAdapter
from samosa.proposals.coupled_proposals import SynceCoupling
from samosa.core.proposal import AdaptiveProposal


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


class SimpleModel(ModelProtocol):
    """Minimal model for testing."""

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = mean
        self.cov = cov
        self.precision = np.linalg.inv(cov)

    def __call__(self, params: np.ndarray) -> dict:
        diff = params - self.mean
        log_posterior = float(-0.5 * (diff.T @ self.precision @ diff).item())
        return {
            "log_posterior": log_posterior,
            "qoi": params[0:1],
            "cost": 1.0,
        }


@pytest.fixture
def chain_states_with_qoi():
    """ChainState list with qoi for _compute_qoi_differences."""
    np.random.seed(42)
    dim = 2
    N = 50
    positions = np.random.randn(dim, N)
    return [
        ChainState(
            position=positions[:, i : i + 1],
            log_posterior=float(-0.5 * np.sum(positions[:, i] ** 2)),
            qoi=positions[0:1, i : i + 1],
            metadata={"iteration": i + 1},
        )
        for i in range(N)
    ]


@pytest.fixture
def mlmc_levels():
    """Minimal MLMC levels for testing (level 0 and level 1)."""
    mean0 = np.zeros((2, 1))
    cov0 = np.eye(2)
    mean1 = np.ones((2, 1)) * 0.5
    cov1 = np.eye(2) * 0.5

    model0 = SimpleModel(mean0, cov0)
    model1 = SimpleModel(mean1, cov1)

    prop = GaussianRandomWalk(mu=np.zeros((2, 1)), cov=np.eye(2))
    adapter = GlobalAdapter(target_ar=0.44, adapt_end=100)
    adaptive_prop = AdaptiveProposal(base_proposal=prop, adapter=adapter)

    kernel0 = MetropolisHastingsKernel(model=model0, proposal=adaptive_prop)
    coupled_prop = SynceCoupling(adaptive_prop, adaptive_prop)
    kernel1 = CoupledKernelBase(
        coarse_model=model0,
        fine_model=model1,
        coupled_proposal=coupled_prop,
    )

    level0 = MLMCLevel(
        level=0,
        kernel=kernel0,
        n_samples=100,
        initial_position_coarse=None,
        initial_position_fine=np.zeros((2, 1)),
    )
    level1 = MLMCLevel(
        level=1,
        kernel=kernel1,
        n_samples=50,
        initial_position_coarse=np.zeros((2, 1)),
        initial_position_fine=np.zeros((2, 1)),
    )
    return [level0, level1]


# -----------------------------------------------------------------------------
# distribute_level_list_across_ranks
# -----------------------------------------------------------------------------


def test_distribute_level_list_across_ranks_even():
    """Levels distributed evenly across ranks."""
    levels = [0, 1, 2, 3]
    # 4 levels, 2 ranks -> 2 per rank
    assert distribute_level_list_across_ranks(levels, rank=0, size=2) == [0, 1]
    assert distribute_level_list_across_ranks(levels, rank=1, size=2) == [2, 3]


def test_distribute_level_list_across_ranks_remainder():
    """Extra levels go to first ranks."""
    levels = [0, 1, 2, 3, 4]
    # 5 levels, 2 ranks -> 3 and 2
    assert distribute_level_list_across_ranks(levels, rank=0, size=2) == [0, 1, 2]
    assert distribute_level_list_across_ranks(levels, rank=1, size=2) == [3, 4]


def test_distribute_level_list_across_ranks_single_rank():
    """Single rank gets all levels."""
    levels = [0, 1, 2]
    assert distribute_level_list_across_ranks(levels, rank=0, size=1) == [0, 1, 2]


def test_distribute_level_list_across_ranks_more_ranks_than_levels():
    """When size > num_levels, some ranks get empty lists."""
    levels = [0, 1]
    assert distribute_level_list_across_ranks(levels, rank=0, size=4) == [0]
    assert distribute_level_list_across_ranks(levels, rank=1, size=4) == [1]
    assert distribute_level_list_across_ranks(levels, rank=2, size=4) == []
    assert distribute_level_list_across_ranks(levels, rank=3, size=4) == []


# -----------------------------------------------------------------------------
# split_comm_for_levels
# -----------------------------------------------------------------------------


def test_split_comm_for_levels_insufficient_ranks():
    """When size < num_levels, returns original comm and rank."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    sub_comm, sub_rank = split_comm_for_levels(comm, num_levels=size + 2, rank=rank)
    assert sub_comm == comm
    assert sub_rank == rank


def test_split_comm_for_levels_single_rank():
    """Single rank: no splitting, returns comm and 0."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    sub_comm, sub_rank = split_comm_for_levels(comm, num_levels=2, rank=rank)
    # With size=1, size < num_levels so we get (comm, rank)
    assert sub_rank == rank


# -----------------------------------------------------------------------------
# create_mlmc_levels
# -----------------------------------------------------------------------------


def test_create_mlmc_levels_success():
    """create_mlmc_levels produces correct MLMCLevel list."""
    model = SimpleModel(np.zeros((2, 1)), np.eye(2))
    prop = GaussianRandomWalk(mu=np.zeros((2, 1)), cov=np.eye(2))
    kernel = MetropolisHastingsKernel(model=model, proposal=prop)

    kernels = [kernel]
    samples_per_level = [100]
    initial_positions = [(None, np.zeros((2, 1)))]

    levels = create_mlmc_levels(
        kernels=kernels,
        samples_per_level=samples_per_level,
        initial_positions=initial_positions,
    )
    assert len(levels) == 1
    assert levels[0].level == 0
    assert levels[0].kernel is kernel
    assert levels[0].initial_position_fine is not None


def test_create_mlmc_levels_length_mismatch():
    """create_mlmc_levels raises when input lengths differ."""
    model = SimpleModel(np.zeros((2, 1)), np.eye(2))
    prop = GaussianRandomWalk(mu=np.zeros((2, 1)), cov=np.eye(2))
    kernel = MetropolisHastingsKernel(model=model, proposal=prop)

    kernels = [kernel]
    samples_per_level = [100, 200]  # length 2
    initial_positions = [(None, np.zeros((2, 1)))]

    with pytest.raises(ValueError, match="same length"):
        create_mlmc_levels(
            kernels=kernels,
            samples_per_level=samples_per_level,
            initial_positions=initial_positions,
        )


# -----------------------------------------------------------------------------
# MLMCCalculator._compute_qoi_differences
# -----------------------------------------------------------------------------


def test_compute_qoi_differences_level0(chain_states_with_qoi):
    """Level 0: samples_coarse is all None, uses qoi from fine (or position)."""
    calc = MLMCCalculator(output_dir="/tmp/dummy", num_levels=1, print_progress=False)
    samples_coarse = [None] * len(chain_states_with_qoi)
    samples_fine = chain_states_with_qoi

    mean_diff, variance_diff, correlations = calc._compute_qoi_differences(
        samples_coarse, samples_fine, burnin=0.2
    )
    assert mean_diff.ndim == 1
    assert variance_diff.ndim == 1
    assert correlations.ndim == 1
    assert np.all(np.isfinite(mean_diff))
    assert np.all(np.isfinite(variance_diff))


def test_compute_qoi_differences_coupled(chain_states_with_qoi):
    """Level > 0: both coarse and fine have ChainStates with qoi."""
    calc = MLMCCalculator(output_dir="/tmp/dummy", num_levels=1, print_progress=False)
    # Coarse: slight offset from fine
    samples_coarse = [
        ChainState(
            position=s.position + 0.1,
            log_posterior=s.log_posterior,
            qoi=s.qoi + 0.1 if s.qoi is not None else s.position[0:1] + 0.1,
            metadata=s.metadata,
        )
        for s in chain_states_with_qoi
    ]
    samples_fine = chain_states_with_qoi

    mean_diff, variance_diff, correlations = calc._compute_qoi_differences(
        samples_coarse, samples_fine, burnin=0.2
    )
    assert mean_diff.ndim == 1
    assert np.all(np.isfinite(mean_diff))
    # Mean diff should be around -0.1 (fine - coarse)
    np.testing.assert_allclose(mean_diff, -0.1, atol=0.5)


def test_compute_qoi_differences_qoi_none_fallback(chain_states_with_qoi):
    """When qoi is None, falls back to position."""
    calc = MLMCCalculator(output_dir="/tmp/dummy", num_levels=1, print_progress=False)
    states_no_qoi = [
        ChainState(
            position=s.position,
            log_posterior=s.log_posterior,
            qoi=None,
            metadata=s.metadata,
        )
        for s in chain_states_with_qoi
    ]
    samples_coarse = [None] * len(states_no_qoi)
    mean_diff, variance_diff, correlations = calc._compute_qoi_differences(
        samples_coarse, states_no_qoi, burnin=0.2
    )
    assert np.all(np.isfinite(mean_diff))


# -----------------------------------------------------------------------------
# MLMCSampler validation
# -----------------------------------------------------------------------------


def test_mlmc_sampler_validation_level0_requires_single_chain_kernel():
    """Level 0 must use a single-chain kernel (MetropolisHastingsKernel or DelayedRejectionKernel)."""
    model = SimpleModel(np.zeros((2, 1)), np.eye(2))
    prop = GaussianRandomWalk(mu=np.zeros((2, 1)), cov=np.eye(2))
    coupled_prop = SynceCoupling(prop, prop)
    kernel = CoupledKernelBase(
        coarse_model=model,
        fine_model=model,
        coupled_proposal=coupled_prop,
    )

    invalid_level = MLMCLevel(
        level=0,
        kernel=kernel,  # invalid: coupled kernel on level 0
        n_samples=10,
        initial_position_fine=np.zeros((2, 1)),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Level 0 must use a single-chain kernel"):
            MLMCSampler(levels=[invalid_level], output_dir=tmpdir, print_progress=False)


def test_mlmc_sampler_validation_level0_requires_initial_position_fine():
    """Level 0 requires initial_position_fine."""
    model = SimpleModel(np.zeros((2, 1)), np.eye(2))
    prop = GaussianRandomWalk(mu=np.zeros((2, 1)), cov=np.eye(2))
    kernel = MetropolisHastingsKernel(model=model, proposal=prop)

    invalid_level = MLMCLevel(
        level=0,
        kernel=kernel,
        n_samples=10,
        initial_position_fine=None,  # invalid
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Level 0 requires initial_position_fine"):
            MLMCSampler(levels=[invalid_level], output_dir=tmpdir, print_progress=False)


def test_mlmc_sampler_validation_level1_requires_coupled_kernel():
    """Level > 0 must use CoupledKernelBase."""
    model = SimpleModel(np.zeros((2, 1)), np.eye(2))
    prop = GaussianRandomWalk(mu=np.zeros((2, 1)), cov=np.eye(2))
    kernel0 = MetropolisHastingsKernel(model=model, proposal=prop)

    level0 = MLMCLevel(
        level=0,
        kernel=kernel0,
        n_samples=10,
        initial_position_fine=np.zeros((2, 1)),
    )
    invalid_level1 = MLMCLevel(
        level=1,
        kernel=kernel0,  # invalid: single-chain kernel on level 1
        n_samples=10,
        initial_position_coarse=np.zeros((2, 1)),
        initial_position_fine=np.zeros((2, 1)),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Level 1 must use a CoupledKernelBase"):
            MLMCSampler(
                levels=[level0, invalid_level1],
                output_dir=tmpdir,
                print_progress=False,
            )


def test_mlmc_sampler_validation_level1_requires_both_initial_positions():
    """Level > 0 requires both initial_position_coarse and initial_position_fine."""
    model = SimpleModel(np.zeros((2, 1)), np.eye(2))
    prop = GaussianRandomWalk(mu=np.zeros((2, 1)), cov=np.eye(2))
    coupled_prop = SynceCoupling(prop, prop)
    kernel0 = MetropolisHastingsKernel(model=model, proposal=prop)
    kernel1 = CoupledKernelBase(
        coarse_model=model,
        fine_model=model,
        coupled_proposal=coupled_prop,
    )

    level0 = MLMCLevel(
        level=0,
        kernel=kernel0,
        n_samples=10,
        initial_position_fine=np.zeros((2, 1)),
    )
    invalid_level1 = MLMCLevel(
        level=1,
        kernel=kernel1,
        n_samples=10,
        initial_position_coarse=None,  # invalid
        initial_position_fine=np.zeros((2, 1)),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="initial_position_coarse"):
            MLMCSampler(
                levels=[level0, invalid_level1],
                output_dir=tmpdir,
                print_progress=False,
            )


# -----------------------------------------------------------------------------
# MLMCResults
# -----------------------------------------------------------------------------


def test_mlmc_results_dataclass():
    """MLMCResults can be instantiated."""
    results = MLMCResults(
        level_means=[np.array([1.0]), np.array([0.5])],
        level_variances=[np.array([0.1]), np.array([0.2])],
        level_correlations=[np.array([0.9]), np.array([0.85])],
        mlmc_expectation=np.array([1.5]),
        mlmc_variance=np.array([0.003]),
        samples_per_level=[100, 50],
        levels_completed=[0, 1],
        burnin_fraction=0.25,
    )
    assert results.mlmc_expectation.shape == (1,)
    assert len(results.levels_completed) == 2


# -----------------------------------------------------------------------------
# MLMCCalculator integration (with saved samples)
# -----------------------------------------------------------------------------


def test_mlmc_calculator_compute_and_load(chain_states_with_qoi):
    """Compute MLMC estimator from saved samples and load results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create level_0 with single-chain samples
        os.makedirs(f"{tmpdir}/level_0", exist_ok=True)
        with open(f"{tmpdir}/level_0/samples.pkl", "wb") as f:
            pickle.dump(chain_states_with_qoi, f)

        # Create level_1 with coupled samples
        os.makedirs(f"{tmpdir}/level_1", exist_ok=True)
        coarse_states = [
            ChainState(
                position=s.position + 0.05,
                log_posterior=s.log_posterior,
                qoi=s.qoi + 0.05 if s.qoi is not None else s.position[0:1] + 0.05,
                metadata=s.metadata,
            )
            for s in chain_states_with_qoi[:40]
        ]
        fine_states = chain_states_with_qoi[:40]
        with open(f"{tmpdir}/level_1/samples_coarse.pkl", "wb") as f:
            pickle.dump(coarse_states, f)
        with open(f"{tmpdir}/level_1/samples_fine.pkl", "wb") as f:
            pickle.dump(fine_states, f)

        calc = MLMCCalculator(output_dir=tmpdir, num_levels=2, print_progress=False)
        results = calc.compute_mlmc_estimator(burnin_fraction=0.2)

        if MPI.COMM_WORLD.Get_rank() == 0:
            assert results is not None
            assert results.levels_completed == [0, 1]
            assert len(results.level_means) == 2
            assert results.mlmc_expectation.ndim >= 1
            assert os.path.exists(f"{tmpdir}/mlmc_results.pkl")

            loaded = calc.load_mlmc_results()
            assert loaded.levels_completed == results.levels_completed
        else:
            assert results is None


def test_mlmc_calculator_level_dir_not_found():
    """compute_mlmc_estimator raises when level directory is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        calc = MLMCCalculator(output_dir=tmpdir, num_levels=2, print_progress=False)
        with pytest.raises(FileNotFoundError, match="Level directory not found"):
            calc.compute_mlmc_estimator(burnin_fraction=0.2)
