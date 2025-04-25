"""
Unit tests for the SYNCE Coupled kernel.
"""

import pytest
import numpy as np
from samosa.core.state import ChainState
from samosa.core.model import ModelProtocol
from samosa.core.proposal import ProposalProtocol
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

class MockProposal(ProposalProtocol):
    """Mock proposal for testing the SYNCE kernel."""
    
    def __init__(self, cov):
        self.cov = cov
        self.adapt_called = False
        
    def adapt(self, state: ChainState) -> None:
        """Record that adapt was called."""
        self.adapt_called = True

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
    """Return a proposal for the coarse model."""
    return MockProposal(cov=np.array([[2.0, 0.0], [0.0, 2.0]]))

@pytest.fixture
def fine_proposal():
    """Return a proposal for the fine model."""
    return MockProposal(cov=np.array([[1.0, 0.0], [0.0, 1.0]]))

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
    def mock_sample(*args, **kwargs):
        return np.array([[0.1], [0.2]])
    
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
    
    # Check that the positions are different but correlated (due to the common noise)
    # The difference should be due to the different covariance matrices
    position_diff_coarse = proposed_coarse.position - coarse_state.position
    position_diff_fine = proposed_fine.position - fine_state.position
    
    # Verify that the positions are updated according to the proposal covariances
    assert np.abs(position_diff_coarse[0, 0]) > np.abs(position_diff_fine[0, 0])

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
    
    # Since the proposed states are closer to the means, they should have higher posteriors
    assert ar_coarse > 0
    assert ar_fine > 0
    
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
    
    # The acceptance ratios should be less than 1 for positions far from the means
    assert ar_coarse < 1.0
    assert ar_fine < 1.0

def test_synce_kernel_adapt(synce_kernel, coarse_proposal, fine_proposal, coarse_state, fine_state):
    """Test the adapt method of SYNCEKernel."""
    # Verify that the adapt_called flags are initially False
    assert not coarse_proposal.adapt_called
    assert not fine_proposal.adapt_called
    
    # Call the adapt method
    synce_kernel.adapt(coarse_proposal, coarse_state, fine_proposal, fine_state)
    
    # Verify that the adapt_called flags are now True
    assert coarse_proposal.adapt_called
    assert fine_proposal.adapt_called

def test_synce_kernel_dimension_mismatch(synce_kernel, coarse_proposal, fine_proposal, coarse_model, fine_model):
    """Test that an assertion error is raised when dimensions don't match."""
    # Create states with different dimensions
    coarse_state = ChainState(
        position=np.array([[1.0], [0.5]]),
        log_posterior=-1.0,
        metadata={}
    )
    
    fine_state = ChainState(
        position=np.array([[1.0], [0.5], [0.3]]),  # Different dimension
        log_posterior=-1.5,
        metadata={}
    )
    
    # The propose method should raise an assertion error
    with pytest.raises(AssertionError):
        synce_kernel.propose(coarse_proposal, fine_proposal, coarse_state, fine_state)

def test_synce_kernel_integration(synce_kernel, coarse_proposal, fine_proposal, coarse_state, fine_state, monkeypatch):
    """Test the full workflow of the SYNCE kernel."""
    # Mock the random number generator to make the test deterministic
    def mock_sample(*args, **kwargs):
        return np.array([[0.1], [0.1]])
    
    monkeypatch.setattr("samosa.kernels.synce.sample_multivariate_gaussian", mock_sample)
    monkeypatch.setattr("numpy.random.rand", lambda: 0.7)  # For acceptance decision
    
    # Propose new states
    proposed_coarse, proposed_fine = synce_kernel.propose(
        coarse_proposal, fine_proposal, coarse_state, fine_state
    )
    
    # Calculate acceptance ratios
    ar_coarse, ar_fine = synce_kernel.acceptance_ratio(
        coarse_proposal, coarse_state, proposed_coarse,
        fine_proposal, fine_state, proposed_fine
    )
    
    # Accept or reject based on the acceptance ratios
    next_coarse = proposed_coarse if (ar_coarse == 1 or np.random.rand() < ar_coarse) else coarse_state
    next_fine = proposed_fine if (ar_fine == 1 or np.random.rand() < ar_fine) else fine_state
    
    # Update metadata
    next_coarse.metadata['acceptance_probability'] = ar_coarse
    next_fine.metadata['acceptance_probability'] = ar_fine
    
    # Adapt the proposals
    synce_kernel.adapt(coarse_proposal, next_coarse, fine_proposal, next_fine)
    
    # Verify that the adaptation was called
    assert coarse_proposal.adapt_called
    assert fine_proposal.adapt_called
