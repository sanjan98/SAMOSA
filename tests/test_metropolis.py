"""
Unit tests for the Metropolis-Hastings kernel.
"""

from typing import cast

import pytest
import numpy as np

from samosa.kernels.metropolis import MetropolisHastingsKernel
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
    """Metropolis-Hastings kernel with model and proposal."""
    return MetropolisHastingsKernel(model, proposal)


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
def test_mh_initialization(model, proposal):
    """Test initialization of the Metropolis-Hastings kernel."""
    k = MetropolisHastingsKernel(model, proposal)
    assert k.model is model
    assert k.proposal is proposal


def test_mh_initialization_rejects_invalid_model(proposal):
    """Kernel raises if model is not Model."""
    with pytest.raises(ValueError, match="model must be an instance of Model"):
        MetropolisHastingsKernel(cast(Model, None), proposal)


def test_mh_initialization_rejects_invalid_proposal(model):
    """Kernel raises if proposal is not Proposal."""
    with pytest.raises(ValueError, match="proposal must be an instance of Proposal"):
        MetropolisHastingsKernel(model, cast(Proposal, None))


def test_propose(kernel, current_state):
    """propose(state) returns a valid proposed state with model evaluations."""
    np.random.seed(42)
    proposed_state = kernel.propose(current_state)
    assert isinstance(proposed_state, ChainState)
    assert proposed_state.position.shape == current_state.position.shape
    assert proposed_state.log_posterior is not None
    expected_logp = -0.5 * np.sum(proposed_state.position**2)
    assert np.isclose(proposed_state.log_posterior, expected_logp)
    assert proposed_state.metadata is not current_state.metadata


def test_acceptance_ratio_accepted(kernel, current_state):
    """acceptance_ratio returns 1.0 when proposed has higher log-posterior."""
    proposed_position = np.array([[0.1], [0.1]])
    proposed_state = ChainState(
        position=proposed_position,
        log_posterior=-0.5 * np.sum(proposed_position**2),
    )
    ar = kernel.acceptance_ratio(current_state, proposed_state)
    assert ar == 1.0


def test_acceptance_ratio_rejected(kernel, current_state):
    """acceptance_ratio returns exp(check) when proposed is worse."""
    proposed_position = np.array([[2.0], [-1.0]])
    proposed_state = ChainState(
        position=proposed_position,
        log_posterior=-0.5 * np.sum(proposed_position**2),
    )
    ar = kernel.acceptance_ratio(current_state, proposed_state)
    assert 0 <= ar <= 1
    assert np.isfinite(ar)


def test_adapt(kernel, current_state):
    """adapt(proposed) runs without error."""
    np.random.seed(42)
    proposed = kernel.propose(current_state)
    kernel.adapt(proposed)


def test_e2e_chain(kernel, current_state):
    """Run a short chain: propose -> acceptance_ratio -> accept/reject."""
    np.random.seed(42)
    states = [current_state]
    for _ in range(5):
        proposed = kernel.propose(states[-1])
        ar = kernel.acceptance_ratio(states[-1], proposed)
        u = np.random.rand()
        if ar == 1.0 or u < ar:
            states.append(proposed)
        else:
            states.append(states[-1])
        kernel.adapt(states[-1])
    assert len(states) == 6
