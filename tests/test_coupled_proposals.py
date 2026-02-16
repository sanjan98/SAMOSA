import numpy as np
import pytest

from samosa.core.proposal import TransportProposalBase
from samosa.core.state import ChainState
from samosa.proposals.coupled_proposals import (
    IndependentCoupling,
    MaximalCoupling,
    SynceCoupling,
)
from samosa.core.map import TransportMap
from samosa.proposals.gaussianproposal import GaussianRandomWalk, IndependentProposal


class IdentityMap(TransportMap):
    def __init__(self):
        super().__init__(dim=2)

    def forward(self, position: np.ndarray):
        return position, 0.0

    def inverse(self, reference_position: np.ndarray):
        return reference_position, 0.0

    def adapt(self, *args, **kwargs):
        return None


def _state(
    position: np.ndarray, mean: np.ndarray | None = None, cov: np.ndarray | None = None
) -> ChainState:
    metadata = {}
    if mean is not None:
        metadata["mean"] = mean
    if cov is not None:
        metadata["covariance"] = cov
    return ChainState(position=position, log_posterior=0.0, metadata=metadata)


def test_synce_rejects_independent_base_proposal():
    dim = 2
    indep = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    rw = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))

    with pytest.raises(ValueError, match="IndependentProposal is not allowed"):
        SynceCoupling(indep, rw)


def test_synce_sample_pair_uses_shared_common_step(monkeypatch):
    dim = 2
    coarse = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    fine = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=4.0 * np.eye(dim))
    coupling = SynceCoupling(coarse, fine)

    coarse_state = _state(np.array([[1.0], [2.0]]))
    fine_state = _state(np.array([[-1.0], [0.5]]))

    eta = np.array([[0.3], [-0.2]])
    monkeypatch.setattr(
        "samosa.proposals.coupled_proposals.sample_multivariate_gaussian",
        lambda *_args, **_kwargs: eta,
    )

    proposed_coarse, proposed_fine = coupling.sample_pair(coarse_state, fine_state)

    np.testing.assert_allclose(proposed_coarse.position, coarse_state.position + eta)
    np.testing.assert_allclose(proposed_fine.position, fine_state.position + 2.0 * eta)


def test_independent_rejects_random_walk_base_proposal():
    dim = 2
    rw = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    indep = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))

    with pytest.raises(ValueError, match="does not allow GaussianRandomWalk"):
        IndependentCoupling(rw, indep)


def test_independent_transport_common_sampler_guard():
    dim = 2
    base_coarse = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    base_fine = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))

    transport_coarse = TransportProposalBase(base_coarse, IdentityMap())
    transport_fine = TransportProposalBase(base_fine, IdentityMap())
    transport_common_sampler = TransportProposalBase(
        IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim)), IdentityMap()
    )

    with pytest.raises(ValueError, match="must be in reference space"):
        IndependentCoupling(
            proposal_coarse=transport_coarse,
            proposal_fine=transport_fine,
            common_sampler=transport_common_sampler,
        )


def test_independent_builds_common_sampler_from_metadata():
    dim = 2
    proposal_coarse = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    proposal_fine = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    coupling = IndependentCoupling(proposal_coarse, proposal_fine)

    coarse_mean = np.array([[2.0], [0.0]])
    fine_mean = np.array([[0.0], [2.0]])
    coarse_cov = np.array([[2.0, 0.0], [0.0, 3.0]])
    fine_cov = np.array([[4.0, 0.0], [0.0, 1.0]])

    coarse_state = _state(np.zeros((dim, 1)), mean=coarse_mean, cov=coarse_cov)
    fine_state = _state(np.ones((dim, 1)), mean=fine_mean, cov=fine_cov)

    coupling.sample_pair(coarse_state, fine_state)

    expected_mean = 0.5 * (coarse_mean + fine_mean)
    expected_cov = 0.5 * (coarse_cov + fine_cov)
    assert coupling.common_sampler is not None
    np.testing.assert_allclose(coupling.common_sampler.mu, expected_mean)
    np.testing.assert_allclose(coupling.common_sampler.cov, expected_cov)


def test_maximal_coupling_supports_transport_wrapped_proposals(monkeypatch):
    dim = 2
    base_coarse = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    base_fine = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))

    transport_coarse = TransportProposalBase(base_coarse, IdentityMap())
    transport_fine = TransportProposalBase(base_fine, IdentityMap())
    coupling = MaximalCoupling(transport_coarse, transport_fine)

    coarse_state = _state(np.zeros((dim, 1)))
    fine_state = _state(np.zeros((dim, 1)))

    # Force the maximal-coupling accept branch.
    monkeypatch.setattr("numpy.random.uniform", lambda *args, **kwargs: 0.0)

    proposed_coarse, proposed_fine = coupling.sample_pair(coarse_state, fine_state)

    assert isinstance(proposed_coarse, ChainState)
    assert isinstance(proposed_fine, ChainState)
    assert proposed_coarse.reference_position is not None
    assert proposed_fine.reference_position is not None
