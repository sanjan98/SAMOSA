import pytest
import numpy as np
from samosa.kernels.delayedrejection import DelayedRejectionKernel
from samosa.proposals.gaussianproposal import GaussianRandomWalk
from samosa.core.state import ChainState

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
    """Delayed Rejection kernel with mock model."""
    return DelayedRejectionKernel(model, cov_scale=0.5)

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
def test_dr_initialization(model):
    """Test initialization of the Delayed Rejection kernel."""
    kernel = DelayedRejectionKernel(model, cov_scale=0.75)
    
    assert kernel.model == model
    assert kernel.cov_scale == 0.75
    assert kernel.first_stage_state is None

def test_proposestate(kernel, proposal, current_state):
    """Test the internal _proposestate method."""
    # Set a fixed seed for reproducibility
    np.random.seed(42)
    
    # Call the internal method directly
    proposed_state = kernel._proposestate(proposal, current_state)
    
    # Verify the state was properly created
    assert isinstance(proposed_state, ChainState)
    assert proposed_state.position.shape == current_state.position.shape
    assert proposed_state.log_posterior is not None
    
    # Expected log posterior value
    expected_logp = -0.5 * np.sum(proposed_state.position**2)
    assert np.isclose(proposed_state.log_posterior, expected_logp)
    
    # Metadata should be copied, not referenced
    assert proposed_state.metadata == current_state.metadata
    assert proposed_state.metadata is not current_state.metadata

def test_acceptance_ratio(kernel, proposal, current_state):
    """Test the standard acceptance ratio calculation."""
    np.random.seed(42)
    
    # Create a proposed state
    proposed_position = np.array([[0.5], [0.5]])
    proposed_state = ChainState(
        position=proposed_position,
        log_posterior=-0.5 * np.sum(proposed_position**2)  # -0.25
    )
    
    # Monkey patch proposal_logpdf for deterministic results
    def mock_logpdf(_, __):
        return -1.0, -1.0  # Symmetric proposal
    
    original_logpdf = proposal.proposal_logpdf
    proposal.proposal_logpdf = mock_logpdf
    
    # Calculate acceptance ratio
    ar = kernel.acceptance_ratio(proposal, current_state, proposed_state)
    
    # Expected: exp(proposed.log_posterior - current.log_posterior)
    # = exp(-0.25 - (-0.625)) = exp(0.375) ≈ 1.455 > 1, so ar = 1.0
    assert ar == 1.0
    
    # Test with worse proposal
    worse_position = np.array([[2.0], [2.0]])
    worse_state = ChainState(
        position=worse_position,
        log_posterior=-0.5 * np.sum(worse_position**2)  # -4.0
    )
    
    ar_worse = kernel.acceptance_ratio(proposal, current_state, worse_state)
    expected_ar_worse = np.exp(worse_state.log_posterior - current_state.log_posterior)
    assert np.isclose(ar_worse, expected_ar_worse)
    
    # Restore original method
    proposal.proposal_logpdf = original_logpdf

def test_second_stage_acceptance_ratio(kernel, proposal, current_state):
    """Test the delayed rejection second stage acceptance ratio."""
    np.random.seed(42)
    
    # Create two proposed states with worsening log posterior
    first_stage = ChainState(
        position=np.array([[1.5], [-1.0]]),
        log_posterior=-1.625  # Worse than current (-0.625)
    )
    
    second_stage = ChainState(
        position=np.array([[0.7], [-0.3]]),
        log_posterior=-0.29  # Better than current (-0.625)
    )
    
    # Set up the first stage state (would normally be done by propose)
    kernel.first_stage_state = first_stage
    
    # Mock the proposal_logpdf to return deterministic values
    original_logpdf = proposal.proposal_logpdf
    
    def mock_logpdf(x, y):
        # For testing, just return fixed values based on which states are being compared
        if np.array_equal(x.position, current_state.position) and np.array_equal(y.position, first_stage.position):
            return -1.0, -1.0  # current to first stage (symmetric)
        elif np.array_equal(x.position, current_state.position) and np.array_equal(y.position, second_stage.position):
            return -1.5, -1.5  # current to second stage (symmetric)
        elif np.array_equal(x.position, second_stage.position) and np.array_equal(y.position, first_stage.position):
            return -2.0, -2.0  # second to first stage (symmetric)
        else:
            return -3.0, -3.0  # default
    
    proposal.proposal_logpdf = mock_logpdf
    
    # Calculate the second stage acceptance ratio
    ar = kernel._second_stage_acceptance_ratio(proposal, current_state, first_stage, second_stage)
    
    # Must be a valid probability
    assert 0 <= ar <= 1.0
    
    # Restore original method
    proposal.proposal_logpdf = original_logpdf

def test_propose_first_stage_accepted(kernel, proposal, current_state, mocker):
    """Test propose method when first stage is accepted."""
    np.random.seed(1)  # This seed should lead to accepting first stage
    
    # Mock _proposestate to return a better state
    better_state = ChainState(
        position=np.array([[0.2], [0.2]]),
        log_posterior=-0.04,  # Better than current (-0.625)
        metadata=current_state.metadata.copy()
    )
    
    mocker.patch.object(kernel, '_proposestate', return_value=better_state)
    
    # Mock acceptance_ratio to always accept
    mocker.patch.object(kernel, 'acceptance_ratio', return_value=1.0)
    
    # Call propose
    result = kernel.propose(proposal, current_state)
    
    # Verify first stage is stored
    assert kernel.first_stage_state is better_state
    
    # First proposal should be accepted
    assert result is better_state
    
    # Verify ar attribute was set
    assert kernel.ar == 1.0

def test_propose_second_stage(kernel, proposal, current_state, mocker):
    """Test propose method when first stage is rejected but second is tried."""
    np.random.seed(42)  # Control random numbers
    
    # Mock _proposestate to return states with specific posteriors
    first_stage = ChainState(
        position=np.array([[1.5], [-1.0]]),
        log_posterior=-1.625,  # Worse than current (-0.625)
        metadata=current_state.metadata.copy()
    )
    
    second_stage = ChainState(
        position=np.array([[0.7], [-0.3]]),
        log_posterior=-0.29,  # Better than current (-0.625)
        metadata=current_state.metadata.copy()
    )
    
    # Set up mocks with side effects to return different states
    proposestate_mock = mocker.patch.object(kernel, '_proposestate')
    proposestate_mock.side_effect = [first_stage, second_stage]
    
    # Mock acceptance_ratio to reject first stage
    mocker.patch.object(kernel, 'acceptance_ratio', return_value=0.0)
    
    # Mock second_stage_acceptance_ratio to accept
    mocker.patch.object(kernel, '_second_stage_acceptance_ratio', return_value=1.0)
    
    # Call propose
    result = kernel.propose(proposal, current_state)
    
    # First stage should be rejected, second should be accepted
    assert result is second_stage
    
    # Check that _proposestate was called twice
    assert proposestate_mock.call_count == 2

def test_propose_both_stages_rejected(kernel, proposal, current_state, mocker):
    """Test propose method when both stages are rejected."""
    np.random.seed(42)  # Control random numbers
    
    # Mock _proposestate to return states with worse posteriors
    first_stage = ChainState(
        position=np.array([[1.5], [-1.0]]),
        log_posterior=-1.625,  # Worse than current (-0.625)
        metadata=current_state.metadata.copy()
    )
    
    second_stage = ChainState(
        position=np.array([[1.2], [-0.8]]),
        log_posterior=-1.04,  # Still worse than current (-0.625)
        metadata=current_state.metadata.copy()
    )
    
    # Set up mocks with side effects
    proposestate_mock = mocker.patch.object(kernel, '_proposestate')
    proposestate_mock.side_effect = [first_stage, second_stage]
    
    # Mock acceptance_ratio to reject first stage
    mocker.patch.object(kernel, 'acceptance_ratio', return_value=0.0)
    
    # Mock second_stage_acceptance_ratio to reject second stage
    mocker.patch.object(kernel, '_second_stage_acceptance_ratio', return_value=0.0)
    
    # Call propose
    result = kernel.propose(proposal, current_state)
    
    # Both stages should be rejected, return current state
    assert result is current_state
    
    # Check that _proposestate was called twice
    assert proposestate_mock.call_count == 2

def test_adapt(kernel, current_state):
    """Test the adapt method."""
    # Create a mock proposal with adapt method
    class MockAdaptiveProposal:
        def __init__(self):
            self.adapted = False
        def adapt(self, state):
            self.adapted = True
    
    proposal = MockAdaptiveProposal()
    
    # Call adapt
    kernel.adapt(proposal, current_state)
    
    # Verify proposal.adapt was called
    assert proposal.adapted