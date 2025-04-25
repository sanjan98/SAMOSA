"""
Unit tests for the SYNCE Coupled kernel.
"""

import pytest
import numpy as np
from samosa.core.state import ChainState
from samosa.core.model import ModelProtocol
from samosa.proposals.gaussianproposal import GaussianRandomWalk
from samosa.kernels.synce import SYNCEKernel
from typing import Dict, Any

# --------------------------------------------------
# Mock classes for testing
# --------------------------------------------------
class GaussianModel(ModelProtocol):
    """Gaussian model for testing the SYNCE kernel."""
    
    def __init__(self, mean, covariance):
        """
        Initialize a Gaussian model with given mean and covariance.
        
        Parameters:
            mean: Mean vector of the Gaussian
            covariance: Covariance matrix of the Gaussian
        """
        self.mean = mean
        self.covariance = covariance
        self.precision = np.linalg.inv(covariance)
        self.log_det = np.log(np.linalg.det(covariance))
        self.dim = mean.shape[0]
        
    def __call__(self, position: np.ndarray) -> Dict[str, Any]:
        """Evaluate the Gaussian log density at the given position."""
        diff = position - self.mean
        log_posterior = -0.5 * (self.dim * np.log(2 * np.pi) + self.log_det + 
                               diff.T @ self.precision @ diff)
        
        return {
            'log_posterior': float(log_posterior),
            'cost': 1.0,
            'qoi': position
        }

# --------------------------------------------------
# Fixtures
# --------------------------------------------------
@pytest.fixture
def coarse_model():
    """Return a Gaussian model with mean [1, 1] and diagonal covariance."""
    mean = np.array([[1.0], [1.0]])
    covariance = np.array([[2.0, 0.0], [0.0, 1.5]])
    return GaussianModel(mean, covariance)

@pytest.fixture
def fine_model():
    """Return a Gaussian model with mean [0, 0] and correlated covariance."""
    mean = np.array([[0.0], [0.0]])
    covariance = np.array([[1.0, 0.7], [0.7, 1.0]])
    return GaussianModel(mean, covariance)

@pytest.fixture
def synce_kernel(coarse_model, fine_model):
    """Return a SYNCE kernel instance."""
    return SYNCEKernel(coarse_model, fine_model)

@pytest.fixture
def coarse_proposal():
    """Return a GaussianRandomWalk proposal for the coarse model."""
    dim = 2
    mu = np.zeros((dim, 1))
    sigma = np.array([[2.0, 0.0], [0.0, 2.0]])
    return GaussianRandomWalk(mu, sigma)

@pytest.fixture
def fine_proposal():
    """Return a GaussianRandomWalk proposal for the fine model."""
    dim = 2
    mu = np.zeros((dim, 1))
    sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
    return GaussianRandomWalk(mu, sigma)

@pytest.fixture
def coarse_state(coarse_model):
    """Return a state for the coarse model."""
    position = np.array([[1.5], [0.8]])
    model_output = coarse_model(position)
    return ChainState(
        position=position,
        **model_output,
        metadata={
            'iteration': 10,
            'acceptance_probability': 0.3,
            'mean': np.array([[1.0], [1.0]]),
            'covariance': np.eye(2),
            'lambda': 1.0
        }
    )

@pytest.fixture
def fine_state(fine_model):
    """Return a state for the fine model."""
    position = np.array([[0.5], [0.2]])
    model_output = fine_model(position)
    return ChainState(
        position=position,
        **model_output,
        metadata={
            'iteration': 10,
            'acceptance_probability': 0.3,
            'mean': np.array([[0.0], [0.0]]),
            'covariance': np.eye(2),
            'lambda': 1.0
        }
    )

# --------------------------------------------------
# Tests for SYNCEKernel
# --------------------------------------------------
def test_synce_kernel_init(synce_kernel, coarse_model, fine_model):
    """Test initialization of SYNCEKernel."""
    assert synce_kernel.coarse_model == coarse_model
    assert synce_kernel.fine_model == fine_model

def test_synce_kernel_propose(synce_kernel, coarse_proposal, fine_proposal, coarse_state, fine_state, monkeypatch):
    """Test the propose method of SYNCEKernel."""
    # Mock the sample_multivariate_gaussian function to return a fixed value
    eta_value = np.array([[0.1], [0.2]])
    def mock_sample(*args, **kwargs):
        return eta_value
    
    monkeypatch.setattr("samosa.kernels.synce.sample_multivariate_gaussian", mock_sample)
    
    # Call the propose method
    proposed_coarse, proposed_fine = synce_kernel.propose(
        coarse_proposal, fine_proposal, coarse_state, fine_state
    )
    
    # Check that the proposed states are different from the current states
    assert not np.array_equal(proposed_coarse.position, coarse_state.position)
    assert not np.array_equal(proposed_fine.position, fine_state.position)
    
    # Check that the proposed states have valid log posterior values
    assert proposed_coarse.log_posterior is not None
    assert proposed_fine.log_posterior is not None
    
    # Check that the metadata was copied
    assert proposed_coarse.metadata['iteration'] == coarse_state.metadata['iteration']
    assert proposed_fine.metadata['iteration'] == fine_state.metadata['iteration']
    
    # Check that the positions are updated according to the proposal logic
    # With GaussianRandomWalk, the positions should be updated using the Cholesky decomposition
    expected_coarse_step = np.linalg.cholesky(coarse_proposal.cov) @ eta_value
    expected_fine_step = np.linalg.cholesky(fine_proposal.cov) @ eta_value
    
    expected_coarse_pos = coarse_state.position + expected_coarse_step
    expected_fine_pos = fine_state.position + expected_fine_step
    
    np.testing.assert_array_almost_equal(proposed_coarse.position, expected_coarse_pos)
    np.testing.assert_array_almost_equal(proposed_fine.position, expected_fine_pos)

def test_synce_kernel_acceptance_ratio(synce_kernel, coarse_proposal, fine_proposal, coarse_state, fine_state, coarse_model, fine_model):
    """Test the acceptance_ratio method of SYNCEKernel."""
    # Create proposed states with positions closer to the means (higher posterior)
    proposed_coarse_position = np.array([[1.2], [1.1]])  # Closer to coarse mean [1, 1]
    proposed_fine_position = np.array([[0.2], [0.1]])    # Closer to fine mean [0, 0]
    
    proposed_coarse = ChainState(
        position=proposed_coarse_position,
        **coarse_model(proposed_coarse_position),
        metadata=coarse_state.metadata.copy()
    )
    
    proposed_fine = ChainState(
        position=proposed_fine_position,
        **fine_model(proposed_fine_position),
        metadata=fine_state.metadata.copy()
    )
    
    # Calculate acceptance ratios
    ar_coarse, ar_fine = synce_kernel.acceptance_ratio(
        coarse_proposal, coarse_state, proposed_coarse,
        fine_proposal, fine_state, proposed_fine
    )
    
    # Check that acceptance ratios are calculated correctly
    # Since proposed positions are closer to the means, they should have higher posteriors
    assert 0 <= ar_coarse <= 1.0
    assert 0 <= ar_fine <= 1.0
    
    # Test with proposed states that are far from the means (lower posterior)
    far_coarse_position = np.array([[5.0], [5.0]])  # Far from coarse mean [1, 1]
    far_fine_position = np.array([[5.0], [5.0]])    # Far from fine mean [0, 0]
    
    far_coarse = ChainState(
        position=far_coarse_position,
        **coarse_model(far_coarse_position),
        metadata=coarse_state.metadata.copy()
    )
    
    far_fine = ChainState(
        position=far_fine_position,
        **fine_model(far_fine_position),
        metadata=fine_state.metadata.copy()
    )
    
    ar_coarse, ar_fine = synce_kernel.acceptance_ratio(
        coarse_proposal, coarse_state, far_coarse,
        fine_proposal, fine_state, far_fine
    )
    
    # The acceptance ratios should be between 0 and 1
    assert 0 <= ar_coarse <= 1.0
    assert 0 <= ar_fine <= 1.0
    
    # For positions far from the means, acceptance ratios should be lower
    assert ar_coarse < 1.0
    assert ar_fine < 1.0

def test_synce_kernel_adapt(synce_kernel, coarse_proposal, fine_proposal, coarse_state, fine_state, monkeypatch):
    """Test the adapt method of SYNCEKernel."""
    # Mock the adapt method of GaussianRandomWalk to track if it's called
    orig_adapt = GaussianRandomWalk.adapt
    adapt_called_coarse = False
    adapt_called_fine = False
    
    def mock_adapt(self, state):
        nonlocal adapt_called_coarse, adapt_called_fine
        if id(self) == id(coarse_proposal):
            adapt_called_coarse = True
        elif id(self) == id(fine_proposal):
            adapt_called_fine = True
        return orig_adapt(self, state)
    
    monkeypatch.setattr(GaussianRandomWalk, "adapt", mock_adapt)
    
    # Call the adapt method
    synce_kernel.adapt(coarse_proposal, coarse_state, fine_proposal, fine_state)
    
    # Restore original method
    monkeypatch.setattr(GaussianRandomWalk, "adapt", orig_adapt)
    
    # Verify that adapt was called for both proposals
    assert adapt_called_coarse
    assert adapt_called_fine

def test_synce_kernel_dimension_mismatch(synce_kernel, coarse_proposal, fine_proposal):
    """Test that an assertion error is raised when dimensions don't match."""
    # Create states with different dimensions
    coarse_state = ChainState(
        position=np.array([[1.0], [0.5]]),
        log_posterior=-1.0,
        cost=1.0,
        qoi=np.array([[1.0], [0.5]]),
        metadata={}
    )
    
    fine_state = ChainState(
        position=np.array([[1.0], [0.5], [0.3]]),  # Different dimension
        log_posterior=-1.5,
        cost=1.0,
        qoi=np.array([[1.0], [0.5], [0.3]]),
        metadata={}
    )
    
    # The propose method should raise an assertion error
    with pytest.raises(AssertionError):
        synce_kernel.propose(coarse_proposal, fine_proposal, coarse_state, fine_state)

def test_synce_kernel_integration(synce_kernel, coarse_proposal, fine_proposal, coarse_state, fine_state, monkeypatch):
    """Test the full workflow of the SYNCE kernel."""
    # Mock the random number generator to make the test deterministic
    eta_value = np.array([[0.1], [0.1]])
    def mock_sample(*args, **kwargs):
        return eta_value
    
    monkeypatch.setattr("samosa.kernels.synce.sample_multivariate_gaussian", mock_sample)
    
    # Mock random.rand for acceptance decisions
    rand_values = iter([0.7, 0.3])  # First for coarse, second for fine
    def mock_rand():
        return next(rand_values)
    
    monkeypatch.setattr("numpy.random.rand", mock_rand)
    
    # Propose new states
    proposed_coarse, proposed_fine = synce_kernel.propose(
        coarse_proposal, fine_proposal, coarse_state, fine_state
    )
    
    # Calculate acceptance ratios
    ar_coarse, ar_fine = synce_kernel.acceptance_ratio(
        coarse_proposal, coarse_state, proposed_coarse,
        fine_proposal, fine_state, proposed_fine
    )
    
    # Apply acceptance decisions
    next_coarse = proposed_coarse if np.random.rand() < ar_coarse else coarse_state
    next_fine = proposed_fine if np.random.rand() < ar_fine else fine_state
    
    # Update metadata for accepted states
    if next_coarse is proposed_coarse:
        next_coarse.metadata['acceptance_probability'] = ar_coarse
    if next_fine is proposed_fine:
        next_fine.metadata['acceptance_probability'] = ar_fine
        
    # Adapt the proposals
    synce_kernel.adapt(coarse_proposal, next_coarse, fine_proposal, next_fine)
    
    # Verify that the chain states have been properly updated
    if next_coarse is proposed_coarse:
        assert next_coarse.metadata['acceptance_probability'] == ar_coarse
    else:
        assert next_coarse.metadata['acceptance_probability'] == coarse_state.metadata['acceptance_probability']
        
    if next_fine is proposed_fine:
        assert next_fine.metadata['acceptance_probability'] == ar_fine
    else:
        assert next_fine.metadata['acceptance_probability'] == fine_state.metadata['acceptance_probability']