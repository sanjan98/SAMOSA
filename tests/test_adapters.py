import numpy as np
import pytest

from samosa.core.state import ChainState
from samosa.proposals.adapters import GlobalAdapter, HaarioAdapter
from samosa.proposals.gaussianproposal import IndependentProposal


@pytest.fixture
def state():
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


@pytest.fixture
def proposal():
    return IndependentProposal(mu=np.zeros((2, 1)), cov=np.eye(2))


def test_haario_adapter_init_validation():
    with pytest.raises(ValueError):
        HaarioAdapter(scale=0.0)


def test_haario_adapter_updates_mean_only_outside_window(state, proposal):
    adapter = HaarioAdapter(scale=2.38**2 / 2, adapt_start=200, adapt_end=500)
    cov_before = proposal.cov.copy()
    mean_before = state.metadata["mean"].copy()

    adapter.adapt(proposal, state)

    assert not np.allclose(state.metadata["mean"], mean_before)
    assert np.allclose(state.metadata["covariance"], np.eye(2))
    assert np.allclose(proposal.cov, cov_before)
    assert np.allclose(proposal.mu, state.metadata["mean"])


def test_haario_adapter_updates_covariance_inside_window(state, proposal):
    adapter = HaarioAdapter(scale=2.38**2 / 2, adapt_start=10, adapt_end=500)
    cov_before = proposal.cov.copy()

    adapter.adapt(proposal, state)

    assert not np.allclose(proposal.cov, cov_before)
    assert np.all(np.linalg.eigvals(proposal.cov) > 0)


def test_haario_adapter_uses_reference_position_when_present(proposal):
    state = ChainState(
        position=np.array([[10.0], [10.0]]),
        reference_position=np.array([[1.0], [-1.0]]),
        log_posterior=-1.0,
        metadata={
            "iteration": 50,
            "mean": np.zeros((2, 1)),
            "covariance": np.eye(2),
            "lambda": 1.0,
            "acceptance_probability": 0.3,
        },
    )
    adapter = HaarioAdapter(scale=1.0, adapt_start=10, adapt_end=100)
    adapter.adapt(proposal, state)
    assert state.reference_position is not None and state.metadata is not None
    assert np.allclose(state.metadata["mean"], state.reference_position / 50.0)


def test_haario_adapter_raises_without_metadata(proposal):
    adapter = HaarioAdapter(scale=1.0)
    state = ChainState(position=np.zeros((2, 1)), log_posterior=0.0, metadata=None)
    with pytest.raises(ValueError):
        adapter.adapt(proposal, state)


def test_global_adapter_init_validation():
    with pytest.raises(ValueError):
        GlobalAdapter(target_ar=0.0)
    with pytest.raises(ValueError):
        GlobalAdapter(target_ar=0.5, C=0.0)
    with pytest.raises(ValueError):
        GlobalAdapter(target_ar=0.5, alpha=0.0)


def test_global_adapter_updates_lambda_inside_window(state, proposal):
    adapter = GlobalAdapter(target_ar=0.234, adapt_start=10, adapt_end=500, C=1.0, alpha=0.5)
    lambda_before = state.metadata["lambda"]
    state.metadata["acceptance_probability"] = 0.5

    adapter.adapt(proposal, state)

    assert state.metadata["lambda"] > lambda_before
    assert np.allclose(proposal.mu, state.metadata["mean"])
    assert np.allclose(proposal.cov, state.metadata["lambda"] * state.metadata["covariance"])


def test_global_adapter_no_lambda_change_outside_window(state, proposal):
    adapter = GlobalAdapter(target_ar=0.234, adapt_start=200, adapt_end=300)
    lambda_before = state.metadata["lambda"]
    cov_before = proposal.cov.copy()

    adapter.adapt(proposal, state)

    assert state.metadata["lambda"] == lambda_before
    assert np.allclose(proposal.cov, cov_before)


def test_global_adapter_raises_without_metadata(proposal):
    adapter = GlobalAdapter()
    state = ChainState(position=np.zeros((2, 1)), log_posterior=0.0, metadata=None)
    with pytest.raises(ValueError):
        adapter.adapt(proposal, state)