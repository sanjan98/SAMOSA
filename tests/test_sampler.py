import pytest
import numpy as np
import os
import tempfile
import pickle
from unittest.mock import patch

from samosa.core.state import ChainState
from samosa.core.proposal import ProposalBase
from samosa.samplers.single_chain import MCMCsampler


# --------------------------------------------------
# Mock classes
# --------------------------------------------------
class MockModel:
    """Mock model returning log_posterior = -0.5 * sum(x^2)."""

    def __call__(self, params: np.ndarray) -> dict:
        return {"log_posterior": -0.5 * np.sum(params**2)}


class MockKernel:
    """Mock kernel that alternates between accepting and rejecting (new API)."""

    def __init__(self, model):
        self.model = model
        self.call_count = 0
        self.ar = 0.0

    def propose(self, state):
        self.call_count += 1
        position = state.position + 0.1 * np.ones_like(state.position)
        proposed_state = ChainState(
            position=position,
            **self.model(position),
            metadata=state.metadata.copy() if state.metadata else {},
        )
        return proposed_state

    def acceptance_ratio(self, current, proposed):
        self.ar = 0.8 if self.call_count % 2 == 0 else 0.2
        return self.ar

    def adapt(self, state, *, samples=None, force_adapt=False):
        pass


class MockProposal(ProposalBase):
    """Mock proposal that adds a constant step (inherits ProposalBase for kernel checks)."""

    def __init__(self, dim):
        cov = np.eye(dim)
        super().__init__(mu=np.zeros((dim, 1)), cov=cov)
        self.dim = dim
        self.sigma = cov
        self.adapted = False

    def sample(self, state, eta=None):
        position = state.position + 0.1 * np.ones((self.dim, 1))
        return ChainState(position=position)

    def proposal_logpdf(self, current, proposed):
        return 0.0, 0.0

    def adapt(self, state, *, samples=None, force_adapt=False, paired_samples=None):
        self.adapted = True


# --------------------------------------------------
# Fixtures
# --------------------------------------------------
@pytest.fixture
def model():
    return MockModel()


@pytest.fixture
def proposal():
    return MockProposal(dim=2)


@pytest.fixture
def kernel(model):
    return MockKernel(model)


@pytest.fixture
def initial_position():
    return np.zeros((2, 1))


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


# --------------------------------------------------
# Tests
# --------------------------------------------------
def test_mcmc_sampler_initialization(model, kernel, proposal, initial_position):
    """Test that the MCMCsampler initializes correctly."""
    n_iterations = 100
    sampler = MCMCsampler(kernel, proposal, initial_position, n_iterations)

    # Check attributes are set correctly
    assert sampler.dim == 2
    assert sampler.model == model
    assert sampler.kernel == kernel
    assert sampler.proposal == proposal
    assert sampler.n_iterations == n_iterations

    # Check initial state is created correctly
    assert np.allclose(sampler.initial_state.position, initial_position)
    assert np.isclose(sampler.initial_state.log_posterior, 0.0)  # -0.5 * sum(0^2) = 0
    assert sampler.initial_state.metadata["iteration"] == 1


def test_mcmc_sampler_run(model, kernel, proposal, initial_position, temp_dir):
    """Test that the MCMCsampler runs correctly."""
    n_iterations = 10
    sampler = MCMCsampler(kernel, proposal, initial_position, n_iterations)

    # Run the sampler
    sampler.run(temp_dir)

    # Check that samples were saved
    assert os.path.exists(os.path.join(temp_dir, "samples.pkl"))

    # Load samples
    with open(os.path.join(temp_dir, "samples.pkl"), "rb") as f:
        samples = pickle.load(f)

    # Check number of samples
    assert len(samples) == n_iterations

    # Check that samples are ChainState objects
    assert all(isinstance(s, ChainState) for s in samples)

    # Check that metadata was updated
    assert samples[7].metadata["iteration"] == 8
    assert samples[-1].metadata["iteration"] == n_iterations


def test_mcmc_sampler_metropolis(model, proposal, initial_position, temp_dir):
    """Test with Metropolis-Hastings kernel."""
    from samosa.kernels.metropolis import MetropolisHastingsKernel

    kernel = MetropolisHastingsKernel(model, proposal)
    n_iterations = 10
    sampler = MCMCsampler(kernel, proposal, initial_position, n_iterations)

    # Run the sampler
    sampler.run(temp_dir)

    # Check that samples were saved
    assert os.path.exists(os.path.join(temp_dir, "samples.pkl"))


def test_mcmc_sampler_delayed_rejection(model, proposal, initial_position, temp_dir):
    """Test with Delayed Rejection kernel."""
    from samosa.kernels.delayedrejection import DelayedRejectionKernel

    kernel = DelayedRejectionKernel(model, proposal)
    n_iterations = 10
    sampler = MCMCsampler(kernel, proposal, initial_position, n_iterations)

    # Verify that the sampler correctly identified the kernel type
    assert sampler.is_delayed_rejection

    # Run the sampler
    sampler.run(temp_dir)

    # Check that samples were saved
    assert os.path.exists(os.path.join(temp_dir, "samples.pkl"))


def test_acceptance_rate(model, kernel, proposal, initial_position, temp_dir):
    """Test that acceptance rate is tracked."""
    n_iterations = 10
    sampler = MCMCsampler(kernel, proposal, initial_position, n_iterations)

    # Run with deterministic acceptance
    with patch("numpy.random.rand", return_value=0.5):
        sampler.run(temp_dir)

    # Load samples
    with open(os.path.join(temp_dir, "samples.pkl"), "rb") as f:
        samples = pickle.load(f)

    # Check that acceptance probabilities were recorded
    assert all("acceptance_probability" in s.metadata for s in samples)
    # Check that acceptance probabilities are in [0, 1]
    assert all(0 <= s.metadata["acceptance_probability"] <= 1 for s in samples)


def test_single_chain_checkpoint_layout(model, proposal, initial_position, temp_dir):
    """Test that checkpoints use output_dir/samples/ and final samples.pkl at root."""
    from samosa.kernels.metropolis import MetropolisHastingsKernel

    kernel = MetropolisHastingsKernel(model, proposal)
    sampler = MCMCsampler(
        kernel, proposal, initial_position, n_iterations=5, save_iteration=2
    )
    sampler.run(temp_dir)

    assert os.path.exists(os.path.join(temp_dir, "samples.pkl"))
    assert os.path.isdir(os.path.join(temp_dir, "samples"))
    assert os.path.exists(os.path.join(temp_dir, "samples", "samples_2.pkl"))
    assert os.path.exists(os.path.join(temp_dir, "samples", "samples_4.pkl"))


def test_single_chain_model_evaluated_on_propose(proposal, initial_position, temp_dir):
    """Test that the model is evaluated when the kernel proposes (MH kernel)."""
    from samosa.kernels.metropolis import MetropolisHastingsKernel

    class CountingModel:
        """Wrapper that counts __call__ invocations."""

        call_count = 0

        def __call__(self, params):
            CountingModel.call_count += 1
            return {"log_posterior": -0.5 * float(np.sum(params**2))}

    counting_model = CountingModel()
    CountingModel.call_count = 0
    kernel = MetropolisHastingsKernel(counting_model, proposal)
    sampler = MCMCsampler(kernel, proposal, initial_position, n_iterations=3)
    sampler.run(temp_dir)

    # Initial state: 1 eval. Each of 3 iterations: propose does 1 model eval. Total >= 4.
    assert CountingModel.call_count >= 4


def test_single_chain_adapt_called_each_iteration(
    model, proposal, initial_position, temp_dir
):
    """Test that proposal.adapt is called each iteration (via MH kernel)."""
    from samosa.kernels.metropolis import MetropolisHastingsKernel

    kernel = MetropolisHastingsKernel(model, proposal)
    sampler = MCMCsampler(kernel, proposal, initial_position, n_iterations=5)
    with patch("numpy.random.rand", return_value=0.5):
        sampler.run(temp_dir)
    assert proposal.adapted is True


def test_single_chain_run_returns_acceptance_rate(
    model, kernel, proposal, initial_position, temp_dir
):
    """Test that run() returns the acceptance rate."""
    sampler = MCMCsampler(kernel, proposal, initial_position, n_iterations=10)
    with patch("numpy.random.rand", return_value=0.5):
        rate = sampler.run(temp_dir)
    assert rate is not None
    assert 0 <= rate <= 1.0
