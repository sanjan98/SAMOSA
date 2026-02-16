"""
Unit tests for the CoupledChainSampler (coupledMCMCsampler) class.
"""

import os
import shutil
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from samosa.core.state import ChainState
from samosa.core.model import ModelProtocol
from samosa.core.proposal import ProposalBase
from samosa.samplers.coupled_chain import coupledMCMCsampler
from samosa.utils.post_processing import load_coupled_samples


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
        log_posterior = -0.5 * self.scale * float(diff.T @ diff)
        return {
            "log_posterior": log_posterior,
            "prior": -0.25 * self.scale * float(diff.T @ diff),
            "likelihood": -0.25 * self.scale * float(diff.T @ diff),
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


class MockKernel:
    """Mock kernel for testing the coupled sampler (new API: propose/acceptance_ratio/adapt)."""

    def __init__(self, coarse_model, fine_model):
        self.coarse_model = coarse_model
        self.fine_model = fine_model
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

    def adapt(self, proposed_coarse, proposed_fine):
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
def kernel(coarse_model, fine_model):
    """Return a mock kernel."""
    return MockKernel(coarse_model, fine_model)


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
    proposal_coarse,
    proposal_fine,
    initial_position_coarse,
    initial_position_fine,
):
    """Return a coupled MCMC sampler instance."""
    return coupledMCMCsampler(
        kernel=kernel,
        proposal_coarse=proposal_coarse,
        proposal_fine=proposal_fine,
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
    initial_position_coarse = np.array([[0.5], [0.5]])
    initial_position_fine = np.array([[0.2], [0.3], [0.4]])  # Different dimension

    with pytest.raises(ValueError):
        coupledMCMCsampler(
            kernel=kernel,
            proposal_coarse=proposal_coarse,
            proposal_fine=proposal_fine,
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
        assert sample_coarse.metadata["iteration"] == i + 1
        assert sample_fine.metadata["iteration"] == i + 1


@pytest.mark.parametrize("n_iterations", [1, 5, 20])
def test_coupled_sampler_different_iterations(
    kernel,
    proposal_coarse,
    proposal_fine,
    initial_position_coarse,
    initial_position_fine,
    output_dir,
    n_iterations,
):
    """Test the sampler with different numbers of iterations."""
    sampler = coupledMCMCsampler(
        kernel=kernel,
        proposal_coarse=proposal_coarse,
        proposal_fine=proposal_fine,
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
        samples_coarse[i - 1].metadata["test_key"] = "test_value"
        assert "test_key" not in samples_coarse[i].metadata


def test_coupled_sampler_checkpoint_layout(
    kernel,
    proposal_coarse,
    proposal_fine,
    initial_position_coarse,
    initial_position_fine,
    output_dir,
    monkeypatch,
):
    """Test that checkpoints use output_dir/samples/ and final .pkl at root."""
    monkeypatch.setattr("numpy.random.rand", lambda: 0.5)
    sampler = coupledMCMCsampler(
        kernel=kernel,
        proposal_coarse=proposal_coarse,
        proposal_fine=proposal_fine,
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
    sampler = coupledMCMCsampler(
        kernel=kernel,
        proposal_coarse=proposal_coarse,
        proposal_fine=proposal_fine,
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
