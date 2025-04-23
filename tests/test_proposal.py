import pytest
import numpy as np
from proposals.gaussianproposal import (
    GaussianRandomWalk,
    IndependentProposal,
)
from core.state import ChainState

# --------------------------------------------------
# Fixtures (Test Data Setup)
# --------------------------------------------------
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

@pytest.fixture
def gaussian_rw():
    """Gaussian random walk proposal with identity covariance."""
    return GaussianRandomWalk(mu=np.zeros((2, 1)), sigma=np.eye(2))

@pytest.fixture
def independent_proposal():
    """Independent proposal centered at origin."""
    return IndependentProposal(mu=np.zeros((2, 1)), sigma=2.0 * np.eye(2))

# --------------------------------------------------
# Gaussian Random Walk Tests
# --------------------------------------------------
def test_gaussian_rw_init(gaussian_rw):
    """Test initialization of GaussianRandomWalk."""
    assert gaussian_rw.mu.shape == (2, 1)
    assert gaussian_rw.cov.shape == (2, 2)
    assert np.allclose(gaussian_rw.mu, np.zeros((2, 1)))
    assert np.allclose(gaussian_rw.cov, np.eye(2))

def test_gaussian_rw_sample(gaussian_rw, current_state):
    """Test that GaussianRandomWalk samples correctly."""
    # Sample multiple times to check randomness
    samples = [gaussian_rw.sample(current_state).position for _ in range(100)]
    for sample in samples:
        assert sample.shape == (2, 1)

    samples = np.array(samples).squeeze()
    
    # Check shape
    assert samples.shape == (100, 2)
    
    # Check that samples are different
    assert len(np.unique(samples, axis=0)) > 1
    
    # Check that samples are centered around current state (approximately)
    assert np.allclose(np.mean(samples, axis=0), current_state.position.flatten(), atol=0.1)

def test_gaussian_rw_logpdf(gaussian_rw, current_state):
    """Test proposal_logpdf for GaussianRandomWalk."""
    # Create proposed state with a known step
    step = np.array([[0.5], [0.5]])
    proposed_state = ChainState(position=current_state.position + step)
    
    # Calculate forward and reverse densities
    logq_fwd, logq_rev = gaussian_rw.proposal_logpdf(current_state, proposed_state)
    
    # For random walk, forward and reverse densities should be equal
    assert np.isclose(logq_fwd, logq_rev, atol=1e-10)
    
    # Check against manual calculation
    expected_log_density = -0.5 * (step.T @ step).item() - np.log(2 * np.pi)
    assert np.isclose(logq_fwd, expected_log_density, atol=1e-6)

# --------------------------------------------------
# Independent Proposal Tests
# --------------------------------------------------
def test_independent_init(independent_proposal):
    """Test initialization of IndependentProposal."""
    assert independent_proposal.mu.shape == (2, 1)
    assert independent_proposal.cov.shape == (2, 2)
    assert np.allclose(independent_proposal.mu, np.zeros((2, 1)))
    assert np.allclose(independent_proposal.cov, 2.0 * np.eye(2))

def test_independent_sample(independent_proposal, current_state):
    """Test sampling from IndependentProposal."""
    # Sample multiple times
    samples = [independent_proposal.sample(current_state).position for _ in range(100)]
    # Assert size of each sample
    for sample in samples:
        assert sample.shape == (2, 1)

    samples = np.array(samples).squeeze()
    
    # Check shape
    assert samples.shape == (100, 2)
    
    # Check randomness
    assert len(np.unique(samples, axis=0)) > 1
    
    # Check samples are centered around proposal mean (approximately)
    # With 50 samples, we should be within 0.5 of the mean
    assert np.allclose(np.mean(samples, axis=0), independent_proposal.mu.flatten(), atol=0.1)
    
    # Check variance is approximately as specified
    # With 50 samples, we should be within 50% of the true variance
    var = np.var(samples, axis=0)
    expected_var = np.diag(independent_proposal.cov).flatten()
    assert np.all((var > 0.5 * expected_var) & (var < 1.5 * expected_var))

def test_independent_logpdf(independent_proposal):
    """Test proposal_logpdf for IndependentProposal."""
    # Create two different states
    state1 = ChainState(position=np.array([[0.0], [0.0]]))
    state2 = ChainState(position=np.array([[1.0], [1.0]]))
    
    # Calculate forward and reverse densities
    logq_fwd, logq_rev = independent_proposal.proposal_logpdf(state1, state2)
    
    # For independent proposal, densities depend only on target position
    # Forward: p(state2 | anything)
    # Reverse: p(state1 | anything)
    
    # Manual calculations
    expected_logq_fwd = -0.5 * (state2.position.T @ np.linalg.inv(independent_proposal.cov) @ state2.position).item() - np.log(2 * np.pi * np.sqrt(np.linalg.det(independent_proposal.cov)))
    expected_logq_rev = -0.5 * (state1.position.T @ np.linalg.inv(independent_proposal.cov) @ state1.position).item() - np.log(2 * np.pi * np.sqrt(np.linalg.det(independent_proposal.cov)))
    
    # Check against expected
    assert np.isclose(logq_fwd, expected_logq_fwd, atol=1e-6)
    assert np.isclose(logq_rev, expected_logq_rev, atol=1e-6)
    
    # Reversing the order should swap the densities
    logq_rev2, logq_fwd2 = independent_proposal.proposal_logpdf(state2, state1)
    assert np.isclose(logq_fwd, logq_fwd2, atol=1e-6)
    assert np.isclose(logq_rev, logq_rev2, atol=1e-6)