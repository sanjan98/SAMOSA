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
from samosa.proposals.gaussianproposal import (
    GaussianRandomWalk,
    IndependentProposal,
    PreCrankNicolson,
)


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


def test_synce_invalid_omega_raises():
    dim = 2
    coarse = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    fine = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    with pytest.raises(ValueError, match="omega must be in"):
        SynceCoupling(coarse, fine, omega=1.1)


def test_synce_with_pcn_bases_initializes_for_pure_synce():
    dim = 2
    mu = np.zeros((dim, 1))
    cov = np.eye(dim)
    coarse = PreCrankNicolson(mu=mu, cov=cov, beta=0.2)
    fine = PreCrankNicolson(mu=mu, cov=cov, beta=0.3)
    coupling = SynceCoupling(coarse, fine, omega=0.0)
    assert isinstance(coupling, SynceCoupling)


def test_synce_omega_one_resyncs_to_common_point():
    dim = 2
    coarse = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    fine = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=4.0 * np.eye(dim))
    coupling = SynceCoupling(coarse, fine, omega=1.0)

    coarse_state = _state(np.array([[1.0], [2.0]]))
    fine_state = _state(np.array([[-1.0], [0.5]]))
    proposed_coarse, proposed_fine = coupling.sample_pair(coarse_state, fine_state)

    np.testing.assert_allclose(proposed_coarse.position, proposed_fine.position)
    assert coupling.common_sampler is not None


def test_synce_omega_one_with_transport_shares_reference():
    dim = 2
    coarse = TransportProposalBase(
        GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim)), IdentityMap()
    )
    fine = TransportProposalBase(
        GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim)), IdentityMap()
    )
    coupling = SynceCoupling(coarse, fine, omega=1.0)

    coarse_state = _state(np.zeros((dim, 1)))
    fine_state = _state(np.ones((dim, 1)))
    proposed_coarse, proposed_fine = coupling.sample_pair(coarse_state, fine_state)

    assert proposed_coarse.reference_position is not None
    assert proposed_fine.reference_position is not None
    np.testing.assert_allclose(
        proposed_coarse.reference_position, proposed_fine.reference_position
    )


def test_independent_accepts_random_walk_and_independent_bases():
    dim = 2
    rw = GaussianRandomWalk(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    indep = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    coupling = IndependentCoupling(rw, indep)
    assert coupling.common_sampler is not None


def test_independent_accepts_pcn_bases():
    dim = 2
    mu = np.zeros((dim, 1))
    cov = np.eye(dim)
    pcn_c = PreCrankNicolson(mu=mu, cov=cov, beta=0.15)
    pcn_f = PreCrankNicolson(mu=mu, cov=cov, beta=0.25)
    coupling = IndependentCoupling(pcn_c, pcn_f)
    coarse_state = _state(np.zeros((dim, 1)))
    fine_state = _state(np.ones((dim, 1)))
    proposed_coarse, proposed_fine = coupling.sample_pair(coarse_state, fine_state)
    assert coupling.common_sampler is not None
    np.testing.assert_allclose(proposed_coarse.position, proposed_fine.position)


def test_independent_builds_common_sampler_from_proposals():
    dim = 2
    coarse_mu = np.array([[2.0], [0.0]])
    fine_mu = np.array([[0.0], [2.0]])
    coarse_cov = np.array([[2.0, 0.0], [0.0, 3.0]])
    fine_cov = np.array([[4.0, 0.0], [0.0, 1.0]])
    proposal_coarse = IndependentProposal(mu=coarse_mu, cov=coarse_cov)
    proposal_fine = IndependentProposal(mu=fine_mu, cov=fine_cov)
    coupling = IndependentCoupling(proposal_coarse, proposal_fine)

    coarse_state = _state(np.zeros((dim, 1)))
    fine_state = _state(np.ones((dim, 1)))

    proposed_coarse, proposed_fine = coupling.sample_pair(coarse_state, fine_state)

    expected_mean = 0.5 * (coarse_mu + fine_mu)
    expected_cov = 0.5 * (coarse_cov + fine_cov)
    assert coupling.common_sampler is not None
    np.testing.assert_allclose(coupling.common_sampler.mu, expected_mean)
    np.testing.assert_allclose(coupling.common_sampler.cov, expected_cov)
    np.testing.assert_allclose(proposed_coarse.position, proposed_fine.position)


def test_independent_transport_shared_reference_maps_to_both_chains():
    dim = 2
    base_coarse = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    base_fine = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    coupling = IndependentCoupling(
        TransportProposalBase(base_coarse, IdentityMap()),
        TransportProposalBase(base_fine, IdentityMap()),
    )

    coarse_state = _state(np.zeros((dim, 1)))
    fine_state = _state(np.ones((dim, 1)))
    proposed_coarse, proposed_fine = coupling.sample_pair(coarse_state, fine_state)

    assert proposed_coarse.reference_position is not None
    assert proposed_fine.reference_position is not None
    np.testing.assert_allclose(
        proposed_coarse.reference_position, proposed_fine.reference_position
    )


def test_independent_refreshes_common_sampler_from_adapted_proposals():
    dim = 2
    proposal_coarse = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    proposal_fine = IndependentProposal(mu=np.zeros((dim, 1)), cov=np.eye(dim))
    coupling = IndependentCoupling(proposal_coarse, proposal_fine)

    # Mimic adaptation updating underlying proposal parameters.
    proposal_coarse.update_parameters(
        mu=np.array([[2.0], [0.0]]), cov=np.array([[2.0, 0.0], [0.0, 2.0]])
    )
    proposal_fine.update_parameters(
        mu=np.array([[0.0], [2.0]]), cov=np.array([[4.0, 0.0], [0.0, 1.0]])
    )

    coarse_state = _state(np.zeros((dim, 1)))
    fine_state = _state(np.zeros((dim, 1)))
    coupling.sample_pair(coarse_state, fine_state)

    np.testing.assert_allclose(coupling.common_sampler.mu, np.array([[1.0], [1.0]]))
    np.testing.assert_allclose(
        coupling.common_sampler.cov, np.array([[3.0, 0.0], [0.0, 1.5]])
    )


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
