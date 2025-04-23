import pytest
import numpy as np
from kernels.metropolis import MetropolisHastingsKernel
from proposals.gaussianproposal import GaussianRandomWalk
from core.state import ChainState

# --------------------------------------------------
# Fixtures
# --------------------------------------------------
@pytest.fixture
def model():
    """Mock model returning log_posterior = -0.5 * sum(x^2)."""
    class MockModel:
        def __call__(self, params: np.ndarray) -> dict:
            return {"log_posterior": -0.5 * np.sum(params**2)}
    return MockModel()

@pytest.fixture
def proposal():
    """Gaussian random walk proposal with fixed parameters."""
    return GaussianRandomWalk(mu=np.zeros((2, 1)), sigma=0.1 * np.eye(2))

@pytest.fixture
def kernel(model):
    """Metropolis-Hastings kernel with mock model."""
    return MetropolisHastingsKernel(model)

@pytest.fixture
def current_state():
    """A sample ChainState for testing proposals."""
    return ChainState(
        position=np.array([[1.0], [-0.5]]),
        log_posterior=-0.625,
        metadata={
            'iteration': 100,
            'mean': np.array([[0.5], [0.5]]),
            'covariance': np.eye(2),
            'lambda': 1.0,
            'acceptance_probability': 0.25
        }
    )

# --------------------------------------------------
# Tests
# --------------------------------------------------
def test_propose(kernel, proposal, current_state):
    """Test that propose() generates valid states with model evaluations."""
    # Set a fixed seed for reproducibility
    np.random.seed(42)
    
    proposed_state = kernel.propose(proposal, current_state)
    
    # Position should differ from current state
    assert not np.allclose(proposed_state.position, current_state.position)
    
    # Model results should be populated
    expected_logp = -0.5 * np.sum(proposed_state.position**2)
    assert np.isclose(proposed_state.log_posterior, expected_logp)
    
    # Metadata should be copied, not referenced
    assert proposed_state.metadata == current_state.metadata
    assert proposed_state.metadata is not current_state.metadata

def test_acceptance_ratio_accepted(kernel, proposal, current_state):
    """Test acceptance ratio when proposed state has higher log-posterior."""
    # Create a proposed state with better log-posterior (closer to 0)
    proposed_position = np.array([[0.1], [0.1]])
    proposed_state = ChainState(
        position=proposed_position,
        log_posterior=-0.5 * np.sum(proposed_position**2)  # -0.01
    )
    
    ar = kernel.acceptance_ratio(proposal, current_state, proposed_state)
    assert ar == 1.0  # Proposed is better, so accept with probability 1

def test_acceptance_ratio_rejected(kernel, proposal, current_state):
    """Test acceptance ratio when proposed state has lower log-posterior."""
    # Create a worse state (farther from 0)
    proposed_position = np.array([[2.0], [-1.0]])
    proposed_state = ChainState(
        position=proposed_position,
        log_posterior=-0.5 * np.sum(proposed_position**2)  # -2.5
    )
    
    # Monkey patch proposal_logpdf to return expected values for this test
    def mock_logpdf(_, __):
        return 0.0, 0.0  # Symmetric proposal with equal forward/reverse density
    
    original_logpdf = proposal.proposal_logpdf
    proposal.proposal_logpdf = mock_logpdf
    
    # Theoretical acceptance ratio
    expected_ar = np.exp(proposed_state.log_posterior - current_state.log_posterior)
    
    ar = kernel.acceptance_ratio(proposal, current_state, proposed_state)
    assert np.isclose(ar, expected_ar, rtol=1e-3)
    
    # Restore original method
    proposal.proposal_logpdf = original_logpdf

def test_acceptance_ratio_asymmetric_proposal(kernel, proposal, current_state):
    """Test acceptance ratio with asymmetric proposal."""
    proposed_position = np.array([[0.5], [0.5]])
    proposed_state = ChainState(
        position=proposed_position,
        log_posterior=-0.5 * np.sum(proposed_position**2)  # -0.25
    )
    
    # Monkey patch proposal_logpdf to return asymmetric densities
    def mock_logpdf(_, __):
        return -1.0, -2.0  # Forward: -1.0, Reverse: -2.0
    
    original_logpdf = proposal.proposal_logpdf
    proposal.proposal_logpdf = mock_logpdf
    
    # Expected acceptance ratio with Metropolis-Hastings correction
    logp_diff = proposed_state.log_posterior - current_state.log_posterior
    logq_diff = -2.0 - (-1.0)  # reverse - forward
    expected_ar = np.exp(logp_diff + logq_diff)
    
    ar = kernel.acceptance_ratio(proposal, current_state, proposed_state)
    assert np.isclose(ar, expected_ar, rtol=1e-3)
    
    # Restore original method
    proposal.proposal_logpdf = original_logpdf

def test_adapt_with_adaptive_proposal(kernel, current_state):
    """Test adaptation with a proposal that supports it."""
    class MockAdaptiveProposal:
        def __init__(self):
            self.adapted = False
        def adapt(self, state):
            self.adapted = True
    
    proposal = MockAdaptiveProposal()
    kernel.adapt(proposal, current_state)
    assert proposal.adapted

def test_adapt_with_non_adaptive_proposal(kernel, current_state):
    """Test adaptation with a proposal that lacks adapt()."""
    class MockProposal:
        pass
    
    proposal = MockProposal()
    # No error should occur
    kernel.adapt(proposal, current_state)

def test_e2e_mh_sample_generation(kernel, proposal, current_state):
    """Test end-to-end sample generation with fixed seed."""
    np.random.seed(42)
    
    # Run a few iterations
    n_iter = 5
    states = [current_state]
    
    for _ in range(n_iter):
        proposed = kernel.propose(proposal, states[-1])
        ar = kernel.acceptance_ratio(proposal, states[-1], proposed)
        u = np.random.rand()
        
        if ar == 1.0 or u < ar:
            states.append(proposed)
        else:
            states.append(states[-1])
    
    # Check that we have the expected number of states
    assert len(states) == n_iter + 1
    
    # Check that some transitions were accepted and some rejected
    positions = np.array([s.position.flatten() for s in states])
    unique_positions = np.unique(positions, axis=0)
    
    # We should have at least 2 unique positions (some accepted)
    assert len(unique_positions) >= 2