import numpy as np
import pytest

from samosa.core.state import ChainState
from samosa.proposals.gaussianproposal import (
    GaussianRandomWalk,
    IndependentProposal,
    PreCrankedNicholson,
)


@pytest.fixture
def current_state():
    return ChainState(
        position=np.array([[1.0], [-0.5]]),
        log_posterior=-0.625,
        metadata={"is_accepted": True, "iteration": 1},
    )


def test_gaussian_random_walk_init_validation():
    with pytest.raises(ValueError):
        GaussianRandomWalk(mu=np.zeros((2, 1)), cov=np.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        GaussianRandomWalk(mu=np.zeros((3, 1)), cov=np.eye(2))


def test_gaussian_random_walk_sample_and_logpdf(current_state):
    proposal = GaussianRandomWalk(mu=np.zeros((2, 1)), cov=np.eye(2))
    proposed = proposal.sample(current_state)
    assert proposed.position.shape == current_state.position.shape

    step = np.array([[0.5], [-0.25]])
    test_state = ChainState(position=current_state.position + step)
    logq_fwd, logq_rev = proposal.proposal_logpdf(current_state, test_state)
    assert np.isclose(logq_fwd, logq_rev, atol=1e-10)


def test_gaussian_random_walk_common_step(current_state):
    proposal = GaussianRandomWalk(mu=np.zeros((2, 1)), cov=np.eye(2))
    common_step = np.array([[0.2], [-0.1]])
    proposed = proposal.sample(current_state, common_step=common_step)
    assert np.allclose(proposed.position, current_state.position + common_step)


def test_gaussian_random_walk_update_parameters_ignores_mu():
    proposal = GaussianRandomWalk(mu=np.zeros((2, 1)), cov=np.eye(2))
    proposal.update_parameters(mu=np.ones((2, 1)), cov=2.0 * np.eye(2))
    assert np.allclose(proposal.mu, np.zeros((2, 1)))
    assert np.allclose(proposal.cov, 2.0 * np.eye(2))


def test_independent_proposal_init_validation():
    with pytest.raises(ValueError):
        IndependentProposal(mu=np.zeros((2, 1)), cov=np.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        IndependentProposal(mu=np.zeros((3, 1)), cov=np.eye(2))


def test_independent_proposal_sample_and_logpdf(current_state):
    proposal = IndependentProposal(mu=np.zeros((2, 1)), cov=2.0 * np.eye(2))
    proposed = proposal.sample(current_state)
    assert proposed.position.shape == current_state.position.shape

    state_a = ChainState(position=np.array([[0.0], [0.0]]))
    state_b = ChainState(position=np.array([[1.0], [1.0]]))
    logq_fwd, logq_rev = proposal.proposal_logpdf(state_a, state_b)
    logq_fwd2, logq_rev2 = proposal.proposal_logpdf(state_b, state_a)
    assert np.isclose(logq_fwd, logq_rev2)
    assert np.isclose(logq_rev, logq_fwd2)


def test_precn_init_validation():
    mu = np.zeros((2, 1))
    cov = np.eye(2)
    with pytest.raises(ValueError):
        PreCrankedNicholson(mu=mu, cov=np.array([1.0, 2.0]), beta=0.2)
    with pytest.raises(ValueError):
        PreCrankedNicholson(mu=np.zeros((3, 1)), cov=cov, beta=0.2)
    with pytest.raises(ValueError):
        PreCrankedNicholson(mu=mu, cov=cov, beta=0.0)
    with pytest.raises(ValueError):
        PreCrankedNicholson(mu=mu, cov=cov, beta=0.2, target_acceptance=1.2)
    with pytest.raises(ValueError):
        PreCrankedNicholson(mu=mu, cov=cov, beta=0.2, beta_min=0.5, beta_max=0.4)


def test_precn_sample_logpdf_and_update(current_state):
    proposal = PreCrankedNicholson(mu=np.zeros((2, 1)), cov=np.eye(2), beta=0.3)
    proposed = proposal.sample(current_state)
    assert proposed.position.shape == current_state.position.shape

    common_step = np.array([[1.0], [0.0]])
    proposed_with_common = proposal.sample(current_state, common_step=common_step)
    assert proposed_with_common.position.shape == current_state.position.shape

    logq_fwd, logq_rev = proposal.proposal_logpdf(current_state, proposed_with_common)
    assert np.isfinite(logq_fwd)
    assert np.isfinite(logq_rev)

    proposal.update_parameters(mu=np.ones((2, 1)), cov=3.0 * np.eye(2))
    # mu is intentionally unchanged for pCN update_parameters
    assert np.allclose(proposal.mu, np.zeros((2, 1)))
    assert np.allclose(proposal.cov, 3.0 * np.eye(2))


def test_precn_adapt_clamps_beta():
    proposal = PreCrankedNicholson(
        mu=np.zeros((2, 1)),
        cov=np.eye(2),
        beta=0.2,
        adjust_rate=5.0,
        beta_min=0.1,
        beta_max=0.3,
    )
    accepted_state = ChainState(position=np.zeros((2, 1)), log_posterior=0.0, metadata={"is_accepted": True})
    rejected_state = ChainState(position=np.zeros((2, 1)), log_posterior=0.0, metadata={"is_accepted": False})

    proposal.adapt(accepted_state)
    assert 0.1 <= proposal.beta <= 0.3

    proposal.adapt(rejected_state)
    assert 0.1 <= proposal.beta <= 0.3