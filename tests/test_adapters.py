"""
Tests for proposal adaptation strategies
"""

import pytest
import numpy as np
from samosa.proposals.adapters import HaarioAdapter, GlobalAdapter
from samosa.proposals.gaussianproposal import GaussianRandomWalk, IndependentProposal
from samosa.core.state import ChainState

# --------------------------------------------------
# Fixtures
# --------------------------------------------------
@pytest.fixture
def current_state():
    """A sample ChainState for testing adaptations."""
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
    """Gaussian random walk proposal for testing adapters."""
    return GaussianRandomWalk(mu=np.zeros((2,1)), sigma=np.eye(2))

@pytest.fixture
def independent_proposal():
    """Independent proposal for testing adapters."""
    return IndependentProposal(mu=np.zeros((2,1)), sigma=np.eye(2))

@pytest.fixture
def haario_adapter():
    """Haario adaptation strategy."""
    return HaarioAdapter(scale=2.38**2/2, adapt_start=50, adapt_end=500)

@pytest.fixture
def global_adapter():
    """Global adaptation strategy."""
    return GlobalAdapter(ar=0.234, adapt_start=50, adapt_end=500)

# --------------------------------------------------
# Haario Adapter Tests
# --------------------------------------------------
def test_haario_init(haario_adapter):
    """Test initialization of HaarioAdapter."""
    assert haario_adapter.scale == 2.38**2/2
    assert haario_adapter.adapt_start == 50
    assert haario_adapter.adapt_end == 500
    assert haario_adapter.eps > 0

def test_haario_before_adaptation_window(haario_adapter, gaussian_rw, current_state):
    """Test that no adaptation occurs before adapt_start."""
    # Save original covariance
    original_cov = gaussian_rw.cov.copy()
    
    # Set iteration to before adapt_start
    current_state.metadata['iteration'] = haario_adapter.adapt_start - 1
    
    # Run adaptation
    haario_adapter.adapt(gaussian_rw, current_state)
    
    # Mean should be set to zero
    assert np.allclose(gaussian_rw.mu, np.zeros((2, 1)))
    # Covariance should remain unchanged
    assert np.allclose(gaussian_rw.cov, original_cov)

def test_haario_after_adaptation_window(haario_adapter, gaussian_rw, current_state):
    """Test that no adaptation occurs after adapt_end."""
    # Save original covariance
    original_cov = gaussian_rw.cov.copy()
    
    # Set iteration to after adapt_end
    current_state.metadata['iteration'] = haario_adapter.adapt_end + 1
    
    # Run adaptation
    haario_adapter.adapt(gaussian_rw, current_state)
    
    # Mean should be set to zero
    assert np.allclose(gaussian_rw.mu, np.zeros((2, 1)))
    # Covariance should remain unchanged
    assert np.allclose(gaussian_rw.cov, original_cov)

def test_haario_during_adaptation_window(haario_adapter, gaussian_rw, current_state):
    """Test that adaptation occurs during the adaptation window."""
    # Save original covariance
    original_cov = gaussian_rw.cov.copy()
    
    # Set iteration within adaptation window
    current_state.metadata['iteration'] = (haario_adapter.adapt_start + haario_adapter.adapt_end) // 2
    
    # Run adaptation
    haario_adapter.adapt(gaussian_rw, current_state)
    
    # Covariance should be updated
    assert not np.allclose(gaussian_rw.cov, original_cov)
    
    # Adapted covariance should still be positive definite
    assert np.all(np.linalg.eigvals(gaussian_rw.cov) > 0)

def test_haario_mean_update(haario_adapter, gaussian_rw, current_state):
    """Test that the mean is updated correctly."""
    # Initial mean
    initial_mean = current_state.metadata['mean'].copy()
    
    # Set iteration within any window as mean update still occures
    current_state.metadata['iteration'] = haario_adapter.adapt_start - 1
    
    # Run adaptation
    haario_adapter.adapt(gaussian_rw, current_state)
    
    # Mean should be updated according to recursive formula
    expected_mean = initial_mean + (current_state.position - initial_mean) / current_state.metadata['iteration']
    assert np.allclose(current_state.metadata['mean'], expected_mean)

def test_haario_with_independent_proposal(haario_adapter, independent_proposal, current_state):
    """Test that HaarioAdapter works with IndependentProposal."""
    # Save original parameters
    original_mu = independent_proposal.mu.copy()
    original_cov = independent_proposal.cov.copy()
    
    # Set iteration within adaptation window
    current_state.metadata['iteration'] = (haario_adapter.adapt_start + haario_adapter.adapt_end) // 2
    
    # Run adaptation
    haario_adapter.adapt(independent_proposal, current_state)
    
    # Mean should be updated
    assert not np.allclose(independent_proposal.mu, original_mu)
    
    # Covariance should be updated
    assert not np.allclose(independent_proposal.cov, original_cov)
    
    # Updated covariance should be positive definite
    assert np.all(np.linalg.eigvals(independent_proposal.cov) > 0)

# --------------------------------------------------
# Global Adapter Tests
# --------------------------------------------------
def test_global_init(global_adapter):
    """Test initialization of GlobalAdapter."""
    assert global_adapter.ar == 0.234
    assert global_adapter.adapt_start == 50
    assert global_adapter.adapt_end == 500
    assert global_adapter.C > 0
    assert global_adapter.alpha > 0
    assert global_adapter.eps > 0

def test_global_before_adaptation_window(global_adapter, gaussian_rw, current_state):
    """Test that no adaptation occurs before adapt_start."""
    # Save original lambda
    original_lambda = current_state.metadata['lambda']

    # Save original covariance
    original_cov = gaussian_rw.cov.copy()
    
    # Set iteration to before adapt_start
    current_state.metadata['iteration'] = global_adapter.adapt_start - 1
    
    # Run adaptation
    global_adapter.adapt(gaussian_rw, current_state)
    
    # Lambda should remain unchanged
    assert current_state.metadata['lambda'] == original_lambda
    # Covariance should remain unchanged
    assert np.allclose(gaussian_rw.cov, original_cov)

def test_global_after_adaptation_window(global_adapter, gaussian_rw, current_state):
    """Test that no adaptation occurs after adapt_end."""
    # Save original lambda
    original_lambda = current_state.metadata['lambda']

    # Save original covariance
    original_cov = gaussian_rw.cov.copy()
    
    # Set iteration to after adapt_end
    current_state.metadata['iteration'] = global_adapter.adapt_end + 1
    
    # Run adaptation
    global_adapter.adapt(gaussian_rw, current_state)
    
    # Lambda should remain unchanged
    assert current_state.metadata['lambda'] == original_lambda
    # Covariance should remain unchanged
    assert np.allclose(gaussian_rw.cov, original_cov)

def test_global_adaptation_low_acceptance(global_adapter, gaussian_rw, current_state):
    """Test adaptation behavior with low acceptance rate."""
    # Set iteration within adaptation window
    current_state.metadata['iteration'] = (global_adapter.adapt_start + global_adapter.adapt_end) // 2
    
    # Set low acceptance rate
    original_lambda = current_state.metadata['lambda']
    current_state.metadata['acceptance_probability'] = 0.1  # below target 0.234
    
    # Run adaptation
    global_adapter.adapt(gaussian_rw, current_state)
    
    # Lambda should decrease to improve acceptance
    assert current_state.metadata['lambda'] < original_lambda

def test_global_adaptation_high_acceptance(global_adapter, gaussian_rw, current_state):
    """Test adaptation behavior with high acceptance rate."""
    # Set iteration within adaptation window
    current_state.metadata['iteration'] = (global_adapter.adapt_start + global_adapter.adapt_end) // 2
    
    # Set high acceptance rate
    original_lambda = current_state.metadata['lambda']
    current_state.metadata['acceptance_probability'] = 0.5  # above target 0.234
    
    # Run adaptation
    global_adapter.adapt(gaussian_rw, current_state)
    
    # Lambda should increase to lower acceptance
    assert current_state.metadata['lambda'] > original_lambda

def test_global_with_independent_proposal(global_adapter, independent_proposal, current_state):
    """Test that GlobalAdapter works with IndependentProposal."""
    # Set iteration within adaptation window
    current_state.metadata['iteration'] = (global_adapter.adapt_start + global_adapter.adapt_end) // 2
    
    # Save original mean and lambda
    original_mu = independent_proposal.mu.copy()
    original_lambda = current_state.metadata['lambda']
    
    # Set high acceptance rate to trigger adaptation
    current_state.metadata['acceptance_probability'] = 0.5  # above target 0.234
    
    # Run adaptation
    global_adapter.adapt(independent_proposal, current_state)
    
    # Mean should be updated
    assert not np.allclose(independent_proposal.mu, original_mu)
    
    # Lambda should be increased to lower acceptance
    assert current_state.metadata['lambda'] > original_lambda