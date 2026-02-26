"""
Unit tests for the CoupledChainSampler (coupledMCMCsampler) class.
"""

import os
import shutil
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import pytest

from samosa.core.state import ChainState
from samosa.core.kernel import CoupledKernelBase
from samosa.core.map import TransportMap
from samosa.core.model import ModelProtocol
from samosa.core.proposal import ProposalBase, TransportProposalBase
from samosa.proposals.coupled_proposals import (
    IndependentCoupling,
    MaximalCoupling,
    SynceCoupling,
)
from samosa.proposals.gaussianproposal import GaussianRandomWalk, IndependentProposal
from samosa.samplers.coupled_chain import coupledMCMCsampler
from samosa.utils.post_processing import (
    get_reference_position_from_states,
    load_coupled_samples,
)


# --------------------------------------------------
# Mock classes for testing
# --------------------------------------------------
class MockModel(ModelProtocol):
    """Mock model for testing the coupled sampler."""

    def __init__(self, scale=1.0, mean=None):
        self.scale = scale
        self.mean = np.zeros((2, 1)) if mean is None else mean

    def __call__(self, position: np.ndarray) -> Dict[str, Any]:
        """Return a mock log posterior value."""
        # Simple quadratic function centered at self.mean
        diff = position - self.mean
        quad = (diff.T @ diff).item()
        log_posterior = -0.5 * self.scale * quad
        return {
            "log_posterior": log_posterior,
            "log_prior": -0.25 * self.scale * quad,
            "log_likelihood": -0.25 * self.scale * quad,
            "cost": self.scale,
            "qoi": position * self.scale,
        }


class MockProposal(ProposalBase):
    """Mock proposal for testing the coupled sampler."""

    def __init__(self, sigma=None):
        sigma = np.eye(2) if sigma is None else sigma
        super().__init__(mu=np.zeros((2, 1)), cov=sigma)
        self.sigma = sigma
        self.adapt_called = False

    def sample(self, current_state: ChainState) -> ChainState:
        """Return a mock sample."""
        # Just return the current state for simplicity in tests
        return ChainState(position=current_state.position.copy())

    def proposal_logpdf(
        self, current_state: ChainState, proposed_state: ChainState
    ) -> Tuple[float, float]:
        """Return mock log probabilities."""
        return 0.0, 0.0

    def adapt(self, state: ChainState) -> None:
        """Record that adapt was called."""
        self.adapt_called = True


class MockCoupledProposal:
    """Minimal coupled proposal for sampler tests (has proposal_coarse, proposal_fine)."""

    def __init__(self, proposal_coarse, proposal_fine):
        self.proposal_coarse = proposal_coarse
        self.proposal_fine = proposal_fine


class MockKernel:
    """Mock kernel for testing the coupled sampler (new API: propose/acceptance_ratio/adapt)."""

    def __init__(self, coarse_model, fine_model, coupled_proposal=None):
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.coupled_proposal = coupled_proposal
        self.propose_called = False
        self.acceptance_ratio_called = False
        self.adapt_called = False

    def propose(self, current_coarse_state, current_fine_state):
        """Return mock proposed states. Returns (proposed_coarse, proposed_fine)."""
        self.propose_called = True
        proposed_coarse_position = current_coarse_state.position + 0.1
        proposed_fine_position = current_fine_state.position + 0.1

        proposed_coarse_state = ChainState(
            position=proposed_coarse_position,
            **self.coarse_model(proposed_coarse_position),
            metadata=current_coarse_state.metadata.copy(),
        )
        proposed_fine_state = ChainState(
            position=proposed_fine_position,
            **self.fine_model(proposed_fine_position),
            metadata=current_fine_state.metadata.copy(),
        )
        return proposed_coarse_state, proposed_fine_state

    def acceptance_ratio(
        self,
        current_coarse,
        proposed_coarse,
        current_fine,
        proposed_fine,
    ):
        """Return mock acceptance ratios."""
        self.acceptance_ratio_called = True
        return 0.7, 0.8

    def adapt(self, proposed_coarse, proposed_fine, *, samples=None):
        """Record that adapt was called."""
        self.adapt_called = True


# --------------------------------------------------
# Fixtures
# --------------------------------------------------
@pytest.fixture
def output_dir():
    """Create and return a temporary output directory."""
    dir_path = "test_coupled_output"
    os.makedirs(dir_path, exist_ok=True)
    yield dir_path
    # Clean up after tests
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


@pytest.fixture
def coarse_model():
    """Return a mock coarse model."""
    return MockModel(scale=0.5, mean=np.array([[1.0], [1.0]]))


@pytest.fixture
def fine_model():
    """Return a mock fine model."""
    return MockModel(scale=1.0, mean=np.array([[0.0], [0.0]]))


@pytest.fixture
def kernel(coarse_model, fine_model, proposal_coarse, proposal_fine):
    """Return a mock kernel with a coupled proposal (sampler gets proposals from kernel)."""
    k = MockKernel(coarse_model, fine_model)
    k.coupled_proposal = MockCoupledProposal(proposal_coarse, proposal_fine)
    return k


@pytest.fixture
def proposal_coarse():
    """Return a mock proposal for the coarse model."""
    return MockProposal(sigma=np.eye(2) * 2.0)


@pytest.fixture
def proposal_fine():
    """Return a mock proposal for the fine model."""
    return MockProposal(sigma=np.eye(2))


@pytest.fixture
def initial_position_coarse():
    """Return an initial position for the coarse chain."""
    return np.array([[0.5], [0.5]])


@pytest.fixture
def initial_position_fine():
    """Return an initial position for the fine chain."""
    return np.array([[0.2], [0.3]])


@pytest.fixture
def coupled_sampler(
    kernel,
    initial_position_coarse,
    initial_position_fine,
):
    """Return a coupled MCMC sampler instance (proposals come from kernel.coupled_proposal)."""
    return coupledMCMCsampler(
        kernel=cast(CoupledKernelBase, kernel),
        initial_position_coarse=initial_position_coarse,
        initial_position_fine=initial_position_fine,
        n_iterations=10,
    )


# --------------------------------------------------
# Tests
# --------------------------------------------------
def test_coupled_sampler_init(
    coupled_sampler,
    coarse_model,
    fine_model,
    kernel,
    proposal_coarse,
    proposal_fine,
    initial_position_coarse,
    initial_position_fine,
):
    """Test initialization of CoupledChainSampler."""
    assert coupled_sampler.dim == initial_position_coarse.shape[0]
    assert coupled_sampler.kernel == kernel
    assert coupled_sampler.proposal_coarse == proposal_coarse
    assert coupled_sampler.proposal_fine == proposal_fine
    assert coupled_sampler.coarse_model == coarse_model
    assert coupled_sampler.fine_model == fine_model
    assert coupled_sampler.n_iterations == 10

    # Check initial states
    assert np.array_equal(
        coupled_sampler.initial_state_coarse.position, initial_position_coarse
    )
    assert np.array_equal(
        coupled_sampler.initial_state_fine.position, initial_position_fine
    )

    # Check metadata
    assert "covariance" in coupled_sampler.initial_state_coarse.metadata
    assert "mean" in coupled_sampler.initial_state_coarse.metadata
    assert "lambda" in coupled_sampler.initial_state_coarse.metadata
    assert "acceptance_probability" in coupled_sampler.initial_state_coarse.metadata
    assert "iteration" in coupled_sampler.initial_state_coarse.metadata


def test_coupled_sampler_dimension_mismatch(kernel, proposal_coarse, proposal_fine):
    """Test that a ValueError is raised when dimensions don't match."""
    kernel.coupled_proposal = MockCoupledProposal(proposal_coarse, proposal_fine)
    initial_position_coarse = np.array([[0.5], [0.5]])
    initial_position_fine = np.array([[0.2], [0.3], [0.4]])  # Different dimension

    with pytest.raises(ValueError):
        coupledMCMCsampler(
            kernel=cast(CoupledKernelBase, kernel),
            initial_position_coarse=initial_position_coarse,
            initial_position_fine=initial_position_fine,
            n_iterations=10,
        )


def test_coupled_sampler_run(coupled_sampler, output_dir, monkeypatch):
    """Test running the coupled sampler."""
    # Mock random number generator for deterministic testing
    monkeypatch.setattr("numpy.random.rand", lambda: 0.5)

    # Run the sampler
    acceptance_rate_coarse, acceptance_rate_fine = coupled_sampler.run(output_dir)

    # Check that the kernel methods were called
    assert coupled_sampler.kernel.propose_called
    assert coupled_sampler.kernel.acceptance_ratio_called
    assert coupled_sampler.kernel.adapt_called

    # Check that the output files were created
    assert os.path.exists(f"{output_dir}/samples_coarse.pkl")
    assert os.path.exists(f"{output_dir}/samples_fine.pkl")

    # Check the acceptance rates (based on our mock kernel returning 0.7 and 0.8)
    assert acceptance_rate_coarse > 0
    assert acceptance_rate_fine > 0

    # With our mock setup and random value of 0.5, we should accept when ar > 0.5
    # Our mock kernel returns ar_coarse=0.7, ar_fine=0.8, so both should be accepted
    assert acceptance_rate_coarse == 1.0  # All proposals accepted
    assert acceptance_rate_fine == 1.0  # All proposals accepted


def test_coupled_sampler_load_samples(coupled_sampler, output_dir, monkeypatch):
    """Test loading samples from files."""
    # Mock random number generator for deterministic testing
    monkeypatch.setattr("numpy.random.rand", lambda: 0.5)

    # Run the sampler to create sample files
    coupled_sampler.run(output_dir)

    # Load the samples
    samples_coarse, samples_fine = load_coupled_samples(output_dir)

    # Check that samples were loaded correctly
    assert len(samples_coarse) == coupled_sampler.n_iterations
    assert len(samples_fine) == coupled_sampler.n_iterations

    # Check that samples are ChainState objects
    assert all(isinstance(s, ChainState) for s in samples_coarse)
    assert all(isinstance(s, ChainState) for s in samples_fine)

    # Check that iteration numbers are set correctly
    for i, (sample_coarse, sample_fine) in enumerate(zip(samples_coarse, samples_fine)):
        meta_c = sample_coarse.metadata or {}
        meta_f = sample_fine.metadata or {}
        assert meta_c.get("iteration") == i + 1
        assert meta_f.get("iteration") == i + 1


@pytest.mark.parametrize("n_iterations", [1, 5, 20])
def test_coupled_sampler_different_iterations(
    kernel,
    initial_position_coarse,
    initial_position_fine,
    output_dir,
    n_iterations,
):
    """Test the sampler with different numbers of iterations."""
    sampler = coupledMCMCsampler(
        kernel=cast(CoupledKernelBase, kernel),
        initial_position_coarse=initial_position_coarse,
        initial_position_fine=initial_position_fine,
        n_iterations=n_iterations,
    )

    # Run the sampler
    sampler.run(output_dir)

    # Load the samples
    samples_coarse, samples_fine = load_coupled_samples(output_dir)

    # Check that the correct number of samples were generated
    assert len(samples_coarse) == n_iterations
    assert len(samples_fine) == n_iterations


def test_coupled_sampler_deep_copy(coupled_sampler, output_dir, monkeypatch):
    """Test that samples are deep copied to avoid reference issues."""
    monkeypatch.setattr("numpy.random.rand", lambda: 0.5)

    coupled_sampler.run(output_dir)
    samples_coarse, samples_fine = load_coupled_samples(output_dir)

    for i in range(1, len(samples_coarse)):
        prev_meta = samples_coarse[i - 1].metadata
        if prev_meta is not None:
            prev_meta["test_key"] = "test_value"
        assert "test_key" not in (samples_coarse[i].metadata or {})


def test_coupled_sampler_checkpoint_layout(
    kernel,
    initial_position_coarse,
    initial_position_fine,
    output_dir,
    monkeypatch,
):
    """Test that checkpoints use output_dir/samples/ and final .pkl at root."""
    monkeypatch.setattr("numpy.random.rand", lambda: 0.5)
    sampler = coupledMCMCsampler(
        kernel=cast(CoupledKernelBase, kernel),
        initial_position_coarse=initial_position_coarse,
        initial_position_fine=initial_position_fine,
        n_iterations=6,
        save_iteration=2,
    )
    sampler.run(output_dir)

    assert os.path.exists(f"{output_dir}/samples_coarse.pkl")
    assert os.path.exists(f"{output_dir}/samples_fine.pkl")
    assert os.path.isdir(f"{output_dir}/samples")
    assert os.path.exists(f"{output_dir}/samples/samples_coarse_2.pkl")
    assert os.path.exists(f"{output_dir}/samples/samples_fine_2.pkl")
    assert os.path.exists(f"{output_dir}/samples/samples_coarse_4.pkl")
    assert os.path.exists(f"{output_dir}/samples/samples_fine_4.pkl")


def test_coupled_sampler_run_returns_acceptance_rates(
    coupled_sampler, output_dir, monkeypatch
):
    """Test that run() returns (acceptance_rate_coarse, acceptance_rate_fine)."""
    monkeypatch.setattr("numpy.random.rand", lambda: 0.5)
    result = coupled_sampler.run(output_dir)
    assert result is not None
    ar_c, ar_f = result
    assert 0 <= ar_c <= 1.0
    assert 0 <= ar_f <= 1.0


def test_coupled_sampler_adapt_called_each_iteration(
    coupled_sampler, output_dir, monkeypatch
):
    """Test that kernel.adapt is called each iteration."""
    monkeypatch.setattr("numpy.random.rand", lambda: 0.5)
    coupled_sampler.run(output_dir)
    assert coupled_sampler.kernel.adapt_called is True


def test_coupled_sampler_proposed_states_have_log_posterior(
    coarse_model,
    fine_model,
    proposal_coarse,
    proposal_fine,
    initial_position_coarse,
    initial_position_fine,
    output_dir,
    monkeypatch,
):
    """Test that after kernel.propose, states have log_posterior (model was evaluated)."""
    monkeypatch.setattr("numpy.random.rand", lambda: 0.5)
    kernel = MockKernel(coarse_model, fine_model)
    kernel.coupled_proposal = MockCoupledProposal(proposal_coarse, proposal_fine)
    sampler = coupledMCMCsampler(
        kernel=cast(CoupledKernelBase, kernel),
        initial_position_coarse=initial_position_coarse,
        initial_position_fine=initial_position_fine,
        n_iterations=2,
    )
    sampler.run(output_dir)
    samples_coarse, samples_fine = load_coupled_samples(output_dir)
    for s in samples_coarse:
        assert s.log_posterior is not None
    for s in samples_fine:
        assert s.log_posterior is not None


# --------------------------------------------------
# E2E with real kernel and coupled proposals
# --------------------------------------------------


class IdentityMap(TransportMap):
    """Identity transport map for E2E tests (e.g. MaximalCoupling + transport)."""

    def __init__(self, dim: int = 2):
        super().__init__(dim=dim)

    def forward(self, position: np.ndarray):
        return position.copy(), 0.0

    def inverse(self, reference_position: np.ndarray):
        return reference_position.copy(), 0.0

    def adapt(self, *args, **kwargs):
        return None


def _run_coupled_e2e(
    coarse_model,
    fine_model,
    coupled_proposal,
    initial_position_coarse: np.ndarray,
    initial_position_fine: np.ndarray,
    output_dir: str,
    n_iterations: int = 30,
    save_iteration: Optional[int] = None,
):
    """Build real CoupledKernelBase, run sampler, load samples. Returns (samples_c, samples_f, ar_c, ar_f)."""
    kernel = CoupledKernelBase(
        coarse_model=coarse_model,
        fine_model=fine_model,
        coupled_proposal=coupled_proposal,
    )
    sampler = coupledMCMCsampler(
        kernel=kernel,
        initial_position_coarse=initial_position_coarse,
        initial_position_fine=initial_position_fine,
        n_iterations=n_iterations,
        save_iteration=save_iteration,
    )
    result = sampler.run(output_dir)
    samples_coarse, samples_fine = load_coupled_samples(output_dir)
    ar_coarse, ar_fine = result if result else (0.0, 0.0)
    return samples_coarse, samples_fine, ar_coarse, ar_fine


def _assert_e2e_basics(samples_coarse, samples_fine, n_iterations, ar_coarse, ar_fine):
    """Common assertions for E2E runs."""
    assert len(samples_coarse) == n_iterations
    assert len(samples_fine) == n_iterations
    for s in samples_coarse:
        assert s.log_posterior is not None
    for s in samples_fine:
        assert s.log_posterior is not None
    assert 0 <= ar_coarse <= 1.0
    assert 0 <= ar_fine <= 1.0


# --- Synce + posteriors ---


def test_e2e_synce_same_gaussian_2d(gaussian_posterior_2d, output_dir):
    """E2E: SynceCoupling + same Gaussian 2d posterior."""
    dim = 2
    grw_c = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    grw_f = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    coupling = SynceCoupling(grw_c, grw_f)
    init_c = np.zeros((dim, 1))
    init_f = np.zeros((dim, 1))
    samples_c, samples_f, ar_c, ar_f = _run_coupled_e2e(
        gaussian_posterior_2d,
        gaussian_posterior_2d,
        coupling,
        init_c,
        init_f,
        output_dir,
        n_iterations=30,
    )
    _assert_e2e_basics(samples_c, samples_f, 30, ar_c, ar_f)
    # Weak check: samples should have spread (Gaussian centered at 0)
    positions_c = np.hstack([s.position for s in samples_c])
    assert np.std(positions_c) > 0


def test_e2e_synce_different_covariances(gaussian_posterior_2d, output_dir):
    """E2E: SynceCoupling with different GRW covariances, same Gaussian 2d."""
    dim = 2
    grw_c = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=2.0 * np.eye(dim))
    grw_f = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=0.5 * np.eye(dim))
    coupling = SynceCoupling(grw_c, grw_f)
    init_c = np.array([[0.5], [0.0]])
    init_f = np.array([[0.0], [0.5]])
    samples_c, samples_f, ar_c, ar_f = _run_coupled_e2e(
        gaussian_posterior_2d,
        gaussian_posterior_2d,
        coupling,
        init_c,
        init_f,
        output_dir,
        n_iterations=30,
    )
    _assert_e2e_basics(samples_c, samples_f, 30, ar_c, ar_f)


def test_e2e_synce_banana_like(banana_like_posterior, output_dir):
    """E2E: SynceCoupling + banana-like posterior (both chains)."""
    dim = 2
    grw_c = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=0.5 * np.eye(dim))
    grw_f = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=0.5 * np.eye(dim))
    coupling = SynceCoupling(grw_c, grw_f)
    init_c = np.array([[0.0], [0.0]])
    init_f = np.array([[0.0], [0.0]])
    samples_c, samples_f, ar_c, ar_f = _run_coupled_e2e(
        banana_like_posterior,
        banana_like_posterior,
        coupling,
        init_c,
        init_f,
        output_dir,
        n_iterations=30,
    )
    _assert_e2e_basics(samples_c, samples_f, 30, ar_c, ar_f)


# --- Independent + posteriors ---


def test_e2e_independent_same_gaussian_2d(gaussian_posterior_2d, output_dir):
    """E2E: IndependentCoupling + same Gaussian 2d (metadata builds common_sampler)."""
    dim = 2
    indep_c = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    indep_f = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    coupling = IndependentCoupling(indep_c, indep_f)
    init_c = np.zeros((dim, 1))
    init_f = np.zeros((dim, 1))
    samples_c, samples_f, ar_c, ar_f = _run_coupled_e2e(
        gaussian_posterior_2d,
        gaussian_posterior_2d,
        coupling,
        init_c,
        init_f,
        output_dir,
        n_iterations=30,
    )
    _assert_e2e_basics(samples_c, samples_f, 30, ar_c, ar_f)


def test_e2e_independent_different_means(
    gaussian_posterior_2d, gaussian_posterior_offset, output_dir
):
    """E2E: IndependentCoupling + coarse vs fine different Gaussian means."""
    dim = 2
    indep_c = IndependentProposal(mu=np.array([[1.0], [-0.5]]), cov=np.eye(dim))
    indep_f = IndependentProposal(mu=np.zeros((dim, 1)), cov=1.5 * np.eye(dim))
    coupling = IndependentCoupling(indep_c, indep_f)
    init_c = np.array([[0.5], [0.0]])
    init_f = np.array([[0.0], [0.5]])
    samples_c, samples_f, ar_c, ar_f = _run_coupled_e2e(
        gaussian_posterior_offset,
        gaussian_posterior_2d,
        coupling,
        init_c,
        init_f,
        output_dir,
        n_iterations=30,
    )
    _assert_e2e_basics(samples_c, samples_f, 30, ar_c, ar_f)


# --- Maximal + posteriors ---


def test_e2e_maximal_same_gaussian_2d(gaussian_posterior_2d, output_dir):
    """E2E: MaximalCoupling(IndependentProposal, IndependentProposal) + same Gaussian 2d."""
    dim = 2
    indep_c = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    indep_f = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    coupling = MaximalCoupling(indep_c, indep_f)
    init_c = np.zeros((dim, 1))
    init_f = np.zeros((dim, 1))
    samples_c, samples_f, ar_c, ar_f = _run_coupled_e2e(
        gaussian_posterior_2d,
        gaussian_posterior_2d,
        coupling,
        init_c,
        init_f,
        output_dir,
        n_iterations=50,
    )
    _assert_e2e_basics(samples_c, samples_f, 50, ar_c, ar_f)


def test_e2e_maximal_transport(gaussian_posterior_2d, output_dir):
    """E2E: MaximalCoupling with TransportProposalBase(IndependentProposal, IdentityMap)."""
    dim = 2
    base_c = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    base_f = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    transport_c = TransportProposalBase(base_c, IdentityMap(dim=dim))
    transport_f = TransportProposalBase(base_f, IdentityMap(dim=dim))
    coupling = MaximalCoupling(transport_c, transport_f)
    init_c = np.zeros((dim, 1))
    init_f = np.zeros((dim, 1))
    samples_c, samples_f, ar_c, ar_f = _run_coupled_e2e(
        gaussian_posterior_2d,
        gaussian_posterior_2d,
        coupling,
        init_c,
        init_f,
        output_dir,
        n_iterations=50,
    )
    _assert_e2e_basics(samples_c, samples_f, 50, ar_c, ar_f)


def test_e2e_synce_transport_reference_positions(gaussian_posterior_2d, output_dir):
    """E2E: SynceCoupling + TransportProposal(IdentityMap); load and check get_reference_position_from_states."""
    dim = 2
    grw_c = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    grw_f = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    transport_c = TransportProposalBase(grw_c, IdentityMap(dim=dim))
    transport_f = TransportProposalBase(grw_f, IdentityMap(dim=dim))
    coupling = SynceCoupling(transport_c, transport_f, omega=1.0)
    n_iterations = 50
    burnin = 0.2
    samples_c, samples_f, ar_c, ar_f = _run_coupled_e2e(
        gaussian_posterior_2d,
        gaussian_posterior_2d,
        coupling,
        np.zeros((dim, 1)),
        np.zeros((dim, 1)),
        output_dir,
        n_iterations=n_iterations,
    )
    _assert_e2e_basics(samples_c, samples_f, n_iterations, ar_c, ar_f)
    ref_c = get_reference_position_from_states(samples_c, burnin=burnin)
    ref_f = get_reference_position_from_states(samples_f, burnin=burnin)
    n_post_burnin = n_iterations - int(n_iterations * burnin)
    assert ref_c.shape == (dim, n_post_burnin)
    assert ref_f.shape == (dim, n_post_burnin)


# --- Restart from checkpoint ---


def test_e2e_restart_from_checkpoint(gaussian_posterior_2d, output_dir):
    """Run 5 iters, load, restart with 5 more; assert iteration continuity (6..10)."""
    dim = 2
    grw_c = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    grw_f = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    coupling = SynceCoupling(grw_c, grw_f)
    kernel = CoupledKernelBase(
        coarse_model=gaussian_posterior_2d,
        fine_model=gaussian_posterior_2d,
        coupled_proposal=coupling,
    )
    init_c = np.zeros((dim, 1))
    init_f = np.zeros((dim, 1))
    sampler1 = coupledMCMCsampler(
        kernel=kernel,
        initial_position_coarse=init_c,
        initial_position_fine=init_f,
        n_iterations=5,
    )
    sampler1.run(output_dir)
    restart_c, restart_f = load_coupled_samples(output_dir)
    assert len(restart_c) == 5
    assert len(restart_f) == 5
    for i, (sc, sf) in enumerate(zip(restart_c, restart_f)):
        assert (sc.metadata or {}).get("iteration") == i + 1
        assert (sf.metadata or {}).get("iteration") == i + 1
    sampler2 = coupledMCMCsampler(
        kernel=kernel,
        initial_position_coarse=init_c,
        initial_position_fine=init_f,
        n_iterations=10,
        restart_coarse=restart_c,
        restart_fine=restart_f,
    )
    sampler2.run(output_dir)
    final_c, final_f = load_coupled_samples(output_dir)
    assert len(final_c) == 10
    assert len(final_f) == 10
    iterations_c = [(s.metadata or {}).get("iteration") for s in final_c]
    iterations_f = [(s.metadata or {}).get("iteration") for s in final_f]
    assert iterations_c == list(range(1, 11))
    assert iterations_f == list(range(1, 11))


# --- Edge cases ---


def test_e2e_n_iterations_one(gaussian_posterior_2d, output_dir):
    """E2E: n_iterations=1 with real kernel + Synce + Gaussian 2d."""
    dim = 2
    grw_c = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    grw_f = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    coupling = SynceCoupling(grw_c, grw_f)
    samples_c, samples_f, ar_c, ar_f = _run_coupled_e2e(
        gaussian_posterior_2d,
        gaussian_posterior_2d,
        coupling,
        np.zeros((dim, 1)),
        np.zeros((dim, 1)),
        output_dir,
        n_iterations=1,
    )
    assert len(samples_c) == 1
    assert len(samples_f) == 1
    assert samples_c[0].log_posterior is not None
    assert samples_f[0].log_posterior is not None
    assert os.path.exists(f"{output_dir}/samples_coarse.pkl")
    assert os.path.exists(f"{output_dir}/samples_fine.pkl")


def test_e2e_checkpoint_layout_real_kernel(gaussian_posterior_2d, output_dir):
    """Checkpoint layout with real kernel + Synce: samples/samples_*_k.pkl."""
    dim = 2
    grw_c = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    grw_f = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    coupling = SynceCoupling(grw_c, grw_f)
    _run_coupled_e2e(
        gaussian_posterior_2d,
        gaussian_posterior_2d,
        coupling,
        np.zeros((dim, 1)),
        np.zeros((dim, 1)),
        output_dir,
        n_iterations=6,
        save_iteration=2,
    )
    assert os.path.exists(f"{output_dir}/samples_coarse.pkl")
    assert os.path.exists(f"{output_dir}/samples_fine.pkl")
    assert os.path.isdir(f"{output_dir}/samples")
    assert os.path.exists(f"{output_dir}/samples/samples_coarse_2.pkl")
    assert os.path.exists(f"{output_dir}/samples/samples_fine_2.pkl")
    assert os.path.exists(f"{output_dir}/samples/samples_coarse_4.pkl")
    assert os.path.exists(f"{output_dir}/samples/samples_fine_4.pkl")


# --- Optional: kernel propose + acceptance_ratio smoke per coupling ---


def _smoke_kernel_one_step(coarse_model, fine_model, coupled_proposal, dim=2):
    """One propose + acceptance_ratio step; no full chain."""
    kernel = CoupledKernelBase(
        coarse_model=coarse_model,
        fine_model=fine_model,
        coupled_proposal=coupled_proposal,
    )
    init_c = ChainState(
        position=np.zeros((dim, 1)),
        log_posterior=0.0,
        metadata={"mean": np.zeros((dim, 1)), "covariance": np.eye(dim)},
    )
    init_f = ChainState(
        position=np.zeros((dim, 1)),
        log_posterior=0.0,
        metadata={"mean": np.zeros((dim, 1)), "covariance": np.eye(dim)},
    )
    prop_c, prop_f = kernel.propose(init_c, init_f)
    ar_c, ar_f = kernel.acceptance_ratio(init_c, prop_c, init_f, prop_f)
    assert 0 <= ar_c and np.isfinite(ar_c)
    assert 0 <= ar_f and np.isfinite(ar_f)
    return ar_c, ar_f


def test_smoke_kernel_synce(gaussian_posterior_2d):
    """One-shot kernel.propose + acceptance_ratio with SynceCoupling."""
    dim = 2
    grw_c = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    grw_f = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    _smoke_kernel_one_step(
        gaussian_posterior_2d,
        gaussian_posterior_2d,
        SynceCoupling(grw_c, grw_f),
        dim=dim,
    )


def test_smoke_kernel_independent(gaussian_posterior_2d):
    """One-shot kernel.propose + acceptance_ratio with IndependentCoupling."""
    dim = 2
    indep_c = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    indep_f = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    _smoke_kernel_one_step(
        gaussian_posterior_2d,
        gaussian_posterior_2d,
        IndependentCoupling(indep_c, indep_f),
        dim=dim,
    )


def test_smoke_kernel_maximal(gaussian_posterior_2d):
    """One-shot kernel.propose + acceptance_ratio with MaximalCoupling."""
    dim = 2
    indep_c = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    indep_f = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    _smoke_kernel_one_step(
        gaussian_posterior_2d,
        gaussian_posterior_2d,
        MaximalCoupling(indep_c, indep_f),
        dim=dim,
    )
