import pytest
import numpy as np
import os
import tempfile
import pickle
from unittest.mock import patch, MagicMock

from core.state import ChainState
from core.model import ModelProtocol
from core.kernel import KernelProtocol
from core.proposal import ProposalProtocol
from samplers.single_chain import MCMCsampler

# --------------------------------------------------
# Mock classes
# --------------------------------------------------
class MockModel:
    """Mock model returning log_posterior = -0.5 * sum(x^2)."""
    def __call__(self, params: np.ndarray) -> dict:
        return {"log_posterior": -0.5 * np.sum(params**2)}

class MockKernel:
    """Mock kernel that alternates between accepting and rejecting."""
    def __init__(self, model):
        self.model = model
        self.call_count = 0
        self.ar = 0.0
        
    def propose(self, proposal, current_state):
        self.call_count += 1
        position = current_state.position + 0.1 * np.ones_like(current_state.position)
        proposed_state = ChainState(position=position, **self.model(position))
        return proposed_state
    
    def acceptance_ratio(self, proposal, current, proposed):
        # Alternate between 0.8 and 0.2 acceptance ratio
        self.ar = 0.8 if self.call_count % 2 == 0 else 0.2
        return self.ar
    
    def adapt(self, proposal, state):
        pass

class MockProposal:
    """Mock proposal that adds a constant step."""
    def __init__(self, dim):
        self.dim = dim
        self.sigma = np.eye(dim)
        self.adapted = False
        
    def sample(self, current_state):
        position = current_state.position + 0.1 * np.ones((self.dim, 1))
        return ChainState(position=position)
    
    def proposal_logpdf(self, current, proposed):
        return 0.0, 0.0
    
    def adapt(self, state):
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
    sampler = MCMCsampler(model, kernel, proposal, initial_position, n_iterations)
    
    # Check attributes are set correctly
    assert sampler.dim == 2
    assert sampler.model == model
    assert sampler.kernel == kernel
    assert sampler.proposal == proposal
    assert sampler.n_iterations == n_iterations
    
    # Check initial state is created correctly
    assert np.allclose(sampler.initial_state.position, initial_position)
    assert np.isclose(sampler.initial_state.log_posterior, 0.0)  # -0.5 * sum(0^2) = 0
    assert sampler.initial_state.metadata['iteration'] == 1

def test_mcmc_sampler_run(model, kernel, proposal, initial_position, temp_dir):
    """Test that the MCMCsampler runs correctly."""
    n_iterations = 10
    sampler = MCMCsampler(model, kernel, proposal, initial_position, n_iterations)
    
    # Run the sampler
    sampler.run(temp_dir)
    
    # Check that samples were saved
    assert os.path.exists(os.path.join(temp_dir, "samples.pkl"))
    
    # Load samples
    with open(os.path.join(temp_dir, "samples.pkl"), "rb") as f:
        samples = pickle.load(f)
    
    # Check number of samples
    assert len(samples) == n_iterations - 1
    
    # Check that samples are ChainState objects
    assert all(isinstance(s, ChainState) for s in samples)
    
    # Check that metadata was updated
    assert samples[-1].metadata['iteration'] == n_iterations - 1

def test_mcmc_sampler_metropolis(model, proposal, initial_position, temp_dir):
    """Test with Metropolis-Hastings kernel."""
    # Use the real kernel for this test
    from kernels.metropolis import MetropolisHastingsKernel
    kernel = MetropolisHastingsKernel(model)
    
    n_iterations = 10
    sampler = MCMCsampler(model, kernel, proposal, initial_position, n_iterations)
    
    # Run the sampler
    sampler.run(temp_dir)
    
    # Check that samples were saved
    assert os.path.exists(os.path.join(temp_dir, "samples.pkl"))

def test_mcmc_sampler_delayed_rejection(model, proposal, initial_position, temp_dir):
    """Test with Delayed Rejection kernel."""
    # Use the real kernel for this test
    from kernels.delayedrejection import DelayedRejectionKernel
    kernel = DelayedRejectionKernel(model)
    
    n_iterations = 10
    sampler = MCMCsampler(model, kernel, proposal, initial_position, n_iterations)
    
    # Verify that the sampler correctly identified the kernel type
    assert sampler.is_delayed_rejection
    
    # Run the sampler
    sampler.run(temp_dir)
    
    # Check that samples were saved
    assert os.path.exists(os.path.join(temp_dir, "samples.pkl"))

def test_acceptance_rate(model, kernel, proposal, initial_position, temp_dir):
    """Test that acceptance rate is tracked."""
    n_iterations = 10
    sampler = MCMCsampler(model, kernel, proposal, initial_position, n_iterations)
    
    # Run with deterministic acceptance
    with patch('numpy.random.rand', return_value=0.5):
        sampler.run(temp_dir)
    
    # Load samples
    with open(os.path.join(temp_dir, "samples.pkl"), "rb") as f:
        samples = pickle.load(f)
    
    # Check that acceptance probabilities were recorded
    assert all('acceptance_probability' in s.metadata for s in samples)
    # Check that acceptance probabilities are in [0, 1]
    assert all(0 <= s.metadata['acceptance_probability'] <= 1 for s in samples)

# Maybe add more complex tests later?