"""
Unit tests for the Delayed Rejection kernel.
"""

from typing import cast
import pytest
import numpy as np

from samosa.kernels.delayedrejection import DelayedRejectionKernel
from samosa.proposals.gaussianproposal import GaussianRandomWalk
from samosa.core.state import ChainState
from samosa.core.model import Model
from samosa.core.proposal import Proposal


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
    return GaussianRandomWalk(mu=np.zeros((2, 1)), cov=0.1 * np.eye(2))


@pytest.fixture
def kernel(model, proposal):
    """Delayed Rejection kernel with model and proposal."""
    return DelayedRejectionKernel(model, proposal, cov_scale=0.5)


@pytest.fixture
def current_state():
    """A sample ChainState for testing."""
    return ChainState(
        position=np.array([[1.0], [-0.5]]),
        log_posterior=-0.625,
        metadata={
            "iteration": 100,
            "mean": np.array([[0.5], [0.5]]),
            "covariance": np.eye(2),
            "lambda": 1.0,
            "acceptance_probability": 0.25,
        },
    )


# --------------------------------------------------
# Tests
# --------------------------------------------------
def test_dr_initialization(model, proposal):
    """Test initialization of the Delayed Rejection kernel."""
    k = DelayedRejectionKernel(model, proposal, cov_scale=0.75)
    assert k.model is model
    assert k.proposal is proposal
    assert k.cov_scale == 0.75
    assert k.first_stage_state is None


def test_dr_initialization_rejects_invalid_model(proposal):
    """Kernel raises if model is not Model."""
    with pytest.raises(ValueError, match="model must be an instance of Model"):
        DelayedRejectionKernel(cast(Model, None), proposal)


def test_dr_initialization_rejects_invalid_proposal(model):
    """Kernel raises if proposal is not Proposal."""
    with pytest.raises(ValueError, match="proposal must be an instance of Proposal"):
        DelayedRejectionKernel(model, cast(Proposal, None))


def test_proposestate(kernel, current_state):
    """_proposestate(state) returns a valid ChainState with model evaluation."""
    np.random.seed(42)
    proposed_state = kernel._proposestate(current_state)
    assert isinstance(proposed_state, ChainState)
    assert proposed_state.position.shape == current_state.position.shape
    assert proposed_state.log_posterior is not None
    expected_logp = -0.5 * np.sum(proposed_state.position**2)
    assert np.isclose(proposed_state.log_posterior, expected_logp)


def test_acceptance_ratio(kernel, current_state):
    """acceptance_ratio(current, proposed) returns a value in [0, 1]."""
    np.random.seed(42)
    proposed_state = kernel._proposestate(current_state)
    ar = kernel.acceptance_ratio(current_state, proposed_state)
    assert 0 <= ar <= 1
    assert np.isfinite(ar)


def test_acceptance_ratio_better_proposal(kernel, current_state):
    """When proposed is better, acceptance_ratio is 1.0 (with symmetric q)."""
    proposed_position = np.array([[0.1], [0.1]])
    proposed_state = ChainState(
        position=proposed_position,
        log_posterior=-0.5 * np.sum(proposed_position**2),
    )
    # Patch proposal_logpdf for symmetric q so AR = min(1, exp(improvement)) = 1
    original = kernel.proposal.proposal_logpdf
    kernel.proposal.proposal_logpdf = lambda c, p: (0.0, 0.0)
    ar = kernel.acceptance_ratio(current_state, proposed_state)
    kernel.proposal.proposal_logpdf = original
    assert ar == 1.0


def test_second_stage_acceptance_ratio(kernel, current_state):
    """_second_stage_acceptance_ratio returns a value in [0, 1]."""
    first_stage = ChainState(
        position=np.array([[1.5], [-1.0]]),
        log_posterior=-1.625,
        metadata=current_state.metadata,
    )
    second_stage = ChainState(
        position=np.array([[0.7], [-0.3]]),
        log_posterior=-0.29,
        metadata=current_state.metadata,
    )
    np.random.seed(42)
    # Scale cov so second-stage logq is consistent
    original_cov = kernel.proposal.cov.copy()
    kernel.proposal.cov = original_cov * kernel.cov_scale
    ar = kernel._second_stage_acceptance_ratio(current_state, first_stage, second_stage)
    kernel.proposal.cov = original_cov
    assert 0 <= ar <= 1
    assert np.isfinite(ar)


def test_second_stage_acceptance_ratio_rejects_inf_posterior(kernel, current_state):
    """_second_stage_acceptance_ratio returns 0 when second_stage.log_posterior is -inf."""
    first_stage = ChainState(
        position=np.array([[1.0], [1.0]]),
        log_posterior=-1.0,
        metadata=current_state.metadata,
    )
    second_stage = ChainState(
        position=np.array([[2.0], [2.0]]),
        log_posterior=-np.inf,
        metadata=current_state.metadata,
    )
    ar = kernel._second_stage_acceptance_ratio(current_state, first_stage, second_stage)
    assert ar == 0.0


def test_propose_returns_single_state(kernel, current_state):
    """propose(state) returns a single ChainState (not a tuple)."""
    np.random.seed(42)
    result = kernel.propose(current_state)
    assert isinstance(result, ChainState)
    assert result.position.shape == current_state.position.shape


def test_propose_first_stage_accepted(kernel, current_state, monkeypatch):
    """When first stage is accepted, propose returns that state."""
    np.random.seed(1)
    better_state = ChainState(
        position=np.array([[0.2], [0.2]]),
        log_posterior=-0.04,
        metadata=current_state.metadata,
    )
    monkeypatch.setattr(kernel, "_proposestate", lambda s: better_state)
    monkeypatch.setattr(kernel, "acceptance_ratio", lambda c, p: 1.0)
    result = kernel.propose(current_state)
    assert result is better_state
    assert kernel.ar == 1.0


def test_propose_second_stage_accepted(kernel, current_state, monkeypatch):
    """When first stage rejected and second accepted, propose returns second state."""
    first_stage = ChainState(
        position=np.array([[1.5], [-1.0]]),
        log_posterior=-1.625,
        metadata=current_state.metadata,
    )
    second_stage = ChainState(
        position=np.array([[0.7], [-0.3]]),
        log_posterior=-0.29,
        metadata=current_state.metadata,
    )
    call_count = [0]

    def proposestate_mock(s):
        call_count[0] += 1
        return [first_stage, second_stage][call_count[0] - 1]

    monkeypatch.setattr(kernel, "_proposestate", proposestate_mock)
    monkeypatch.setattr(kernel, "acceptance_ratio", lambda c, p: 0.0)
    monkeypatch.setattr(kernel, "_second_stage_acceptance_ratio", lambda c, f, s: 1.0)
    np.random.seed(42)
    result = kernel.propose(current_state)
    assert result is second_stage
    assert call_count[0] == 2


def test_propose_both_stages_rejected(kernel, current_state, monkeypatch):
    """When both stages rejected, propose returns current state."""
    first_stage = ChainState(
        position=np.array([[1.5], [-1.0]]),
        log_posterior=-1.625,
        metadata=current_state.metadata,
    )
    second_stage = ChainState(
        position=np.array([[1.2], [-0.8]]),
        log_posterior=-1.04,
        metadata=current_state.metadata,
    )
    call_count = [0]

    def proposestate_mock(s):
        call_count[0] += 1
        return [first_stage, second_stage][call_count[0] - 1]

    monkeypatch.setattr(kernel, "_proposestate", proposestate_mock)
    monkeypatch.setattr(kernel, "acceptance_ratio", lambda c, p: 0.0)
    monkeypatch.setattr(kernel, "_second_stage_acceptance_ratio", lambda c, f, s: 0.0)
    np.random.seed(42)
    result = kernel.propose(current_state)
    assert result is current_state
    assert call_count[0] == 2


def test_adapt(kernel, current_state):
    """adapt(proposed) runs without error."""
    np.random.seed(42)
    proposed = kernel.propose(current_state)
    kernel.adapt(proposed)


def test_e2e_dr_chain(kernel, current_state):
    """Run a short DR chain: propose -> accept/reject -> adapt."""
    np.random.seed(123)
    states = [current_state]
    for _ in range(5):
        proposed = kernel.propose(states[-1])
        # DR kernel stores ar in kernel.ar; for first stage it's from acceptance_ratio
        u = np.random.rand()
        if kernel.ar == 1.0 or u < kernel.ar:
            states.append(proposed)
        else:
            states.append(states[-1])
        kernel.adapt(states[-1])
    assert len(states) == 6
