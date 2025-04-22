import pytest
import numpy as np
from proposals.gaussianproposal import (
    GaussianRandomWalk,
    IndependentProposal,
    HaarioAdaptiveProposal,
    GlobalAdaptiveProposal
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
        log_posterior=-2.3,
        metadata={
            'iteration': 100,
            'mean': np.array([[0.5], [0.5]]),
            'covariance': np.eye(2),
            'lambda': 1.0,
            'acceptance_probability': 0.25
        }
    )

# --------------------------------------------------
# GaussianRandomWalk Tests
# --------------------------------------------------
def test_gaussian_rw_sample_logpdf(current_state):
    """Test that GaussianRandomWalk generates valid proposals."""
    proposal = GaussianRandomWalk(mu=np.zeros(2).reshape(2, 1), sigma=np.eye(2))
    proposed_state = proposal.sample(current_state)
    
    # Check position update
    assert proposed_state.position.shape == (2, 1)
    assert not np.allclose(proposed_state.position, current_state.position)

    logq_fwd, logq_rev = proposal.proposal_logpdf(current_state, proposed_state)
    
    # Symmetric proposal should have logq_fwd ≈ logq_rev
    assert np.isclose(logq_fwd, logq_rev, rtol=1e-3)

# --------------------------------------------------
# IndependentProposal Tests
# --------------------------------------------------
def test_independent_proposal_logpdf():
    """Test logpdf calculation for IndependentProposal."""
    proposal = IndependentProposal(mu=np.array([[1.0], [1.0]]), sigma=np.eye(2))
    current = ChainState(position=np.array([[0.0], [0.0]]))
    proposed = ChainState(position=np.array([[1.0], [1.0]]))
    
    logq_fwd, logq_rev = proposal.proposal_logpdf(current, proposed)
    
    # Forward logpdf should be N(1|1, I) = -0.5*(0)^2 = 0.0
    # Reverse logpdf should be N(0|1, I) = -0.5*(1)^2*2 = -1.0
    assert np.isclose(logq_fwd, 0.0 + np.log(1.0 / (2.0 * np.pi)), atol=1e-6)
    assert np.isclose(logq_rev, -1.0 + np.log(1.0 / (2.0 * np.pi)), atol=1e-6)

# --------------------------------------------------
# HaarioAdaptiveProposal Tests
# --------------------------------------------------
def test_haario_adaptation(current_state):
    """Test covariance adaptation in HaarioAdaptiveProposal."""
    initial_cov = np.eye(2)
    proposal = HaarioAdaptiveProposal(
        mu=np.zeros(2).reshape(2, 1),
        sigma=initial_cov,
        scale=1.0,
        adapt_start=50,
        adapt_end=200,
        eps=1e-6
    )

    # Before adaptation window
    current_state.metadata['iteration'] = 49
    proposal.adapt(current_state)
    assert np.allclose(proposal.cov, initial_cov)  # No adaptation
    
    # During adaptation window
    # Reset current state for the next test
    current_state.metadata['mean'] = np.array([[0.5], [0.5]])
    current_state.metadata['covariance'] = np.eye(2)
    current_state.metadata['iteration'] = 100
    proposal.adapt(current_state)
    assert not np.allclose(proposal.cov, initial_cov)  # Covariance updated
    
    # Check covariance is positive definite
    assert np.all(np.linalg.eigvals(proposal.cov) > 0)

    # Check mean update
    assert np.allclose(current_state.metadata['mean'], np.array([[0.505],[0.49]]), atol=1e-6)  # Updated mean

    # Check covariance update
    # Maybe later? -> I checked one entry seems right

# --------------------------------------------------
# GlobalAdaptiveProposal Tests
# --------------------------------------------------
def test_global_adaptation(current_state):
    """Test step size adaptation in GlobalAdaptiveProposal."""
    proposal = GlobalAdaptiveProposal(
        mu=np.zeros(2).reshape(2, 1),
        sigma=np.eye(2),
        ar=0.234,
        adapt_start=50,
        adapt_end=200,
        C=1.0,
        alpha=0.5,
        eps=1e-6
    )
    
    # Before adaptation
    initial_lambda = 1.0

    # Check mean and covariance
    proposal.adapt(current_state)
    assert np.allclose(current_state.metadata['mean'], np.array([[0.55],[0.4]]), atol=1e-6)  # Updated mean
    # Check covariance is positive definite
    assert np.all(np.linalg.eigvals(current_state.metadata['covariance']) > 0)
    
    # Adapt with acceptance rate below target (0.1 < 0.234)
    current_state.metadata['acceptance_probability'] = 0.1
    proposal.adapt(current_state)
    assert current_state.metadata['lambda'] < initial_lambda  # Decrease step size
    
    # Adapt with acceptance rate above target (0.5 > 0.234)
    current_state.metadata['acceptance_probability'] = 0.5
    proposal.adapt(current_state)
    assert current_state.metadata['lambda'] > initial_lambda  # Increase step size

if __name__ == "__main__":
    cs = ChainState(
        position=np.array([[1.0], [-0.5]]),
        log_posterior=-2.3,
        metadata={
            'iteration': 100,
            'mean': np.array([[0.5], [0.5]]),
            'covariance': np.eye(2),
            'lambda': 1.0,
            'acceptance_probability': 0.25
        }
    )
    # test_haario_adaptation(cs)
    test_global_adaptation(cs)