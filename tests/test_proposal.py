import numpy as np
import pytest

from samosa.core.proposal import (
    AdapterBase,
    AdaptiveProposal,
    CoupledProposalBase,
    ProposalBase,
    TransportProposalBase,
)
from samosa.proposals.maximalproposal import MaximalCoupling
from samosa.core.state import ChainState


class ConcreteProposal(ProposalBase):
    def sample(self, current_state: ChainState, common_step=None) -> ChainState:
        if common_step is None:
            step = np.zeros_like(current_state.position)
        else:
            step = common_step
        return ChainState(position=current_state.position + step)

    def proposal_logpdf(
        self, current_state: ChainState, proposed_state: ChainState
    ) -> tuple[float, float]:
        return 0.0, 0.0


class ConcreteAdapter(AdapterBase):
    def __init__(self):
        super().__init__(adapt_start=1, adapt_end=10, eps=1e-6)
        self.calls = 0

    def adapt(self, proposal: ProposalBase, state: ChainState) -> None:
        self.calls += 1
        proposal.update_parameters(mu=state.position)


class DummyAdapter(AdapterBase):
    def adapt(self, proposal: ProposalBase, state: ChainState) -> None:
        return None


class DummyCoupledProposal(CoupledProposalBase):
    def __init__(self, proposal_coarse: ProposalBase, proposal_fine: ProposalBase):
        super().__init__(proposal_coarse, proposal_fine)

    def sample_pair(self, coarse_state: ChainState, fine_state: ChainState):
        return coarse_state, fine_state

    def proposal_logpdf_pair(
        self,
        current_coarse: ChainState,
        proposed_coarse: ChainState,
        current_fine: ChainState,
        proposed_fine: ChainState,
    ):
        return (0.0, 0.0), (0.0, 0.0)

    def adapt_pair(self, coarse_state: ChainState, fine_state: ChainState) -> None:
        return None


class TrackingProposal(ConcreteProposal):
    def __init__(self, mu: np.ndarray, cov: np.ndarray):
        super().__init__(mu=mu, cov=cov)
        self.adapt_calls = []

    def adapt(
        self,
        state: ChainState,
        *,
        samples: list[ChainState] | None = None,
        force_adapt: bool = False,
        paired_samples: list[ChainState] | None = None,
    ) -> None:
        self.adapt_calls.append(
            {
                "state": state,
                "samples": samples,
                "force_adapt": force_adapt,
                "paired_samples": paired_samples,
            }
        )


class TrackingMap:
    def __init__(self):
        self.adapt_calls = []

    def forward(self, position: np.ndarray):
        return position, 0.0

    def inverse(self, reference_position: np.ndarray):
        return reference_position, 0.0

    def adapt(
        self,
        samples: list[ChainState],
        *,
        force_adapt: bool = False,
        paired_samples: list[ChainState] | None = None,
    ) -> None:
        self.adapt_calls.append(
            {
                "samples": samples,
                "force_adapt": force_adapt,
                "paired_samples": paired_samples,
            }
        )


class InheritedCoupledProposal(CoupledProposalBase):
    def sample_pair(self, coarse_state: ChainState, fine_state: ChainState):
        return coarse_state, fine_state


def test_maximal_coupling_sample_pair_returns_chainstates_for_base_proposals():
    proposal_coarse = ConcreteProposal(mu=np.zeros((2, 1)), cov=np.eye(2))
    proposal_fine = ConcreteProposal(mu=np.zeros((2, 1)), cov=np.eye(2))
    coupling = MaximalCoupling(proposal_coarse, proposal_fine)

    coarse_state = ChainState(
        position=np.array([[0.0], [0.0]]), log_posterior=0.0, metadata={"iteration": 1}
    )
    fine_state = ChainState(
        position=np.array([[1.0], [1.0]]), log_posterior=0.0, metadata={"iteration": 1}
    )

    proposed_coarse, proposed_fine = coupling.sample_pair(coarse_state, fine_state)

    assert isinstance(proposed_coarse, ChainState)
    assert isinstance(proposed_fine, ChainState)
    assert proposed_coarse.position.shape == (2, 1)
    assert proposed_fine.position.shape == (2, 1)


def test_maximal_coupling_supports_transport_wrapped_proposals():
    base_coarse = ConcreteProposal(mu=np.zeros((2, 1)), cov=np.eye(2))
    base_fine = ConcreteProposal(mu=np.zeros((2, 1)), cov=np.eye(2))
    map_coarse = TrackingMap()
    map_fine = TrackingMap()
    transport_coarse = TransportProposalBase(base_coarse, map_coarse)
    transport_fine = TransportProposalBase(base_fine, map_fine)
    coupling = MaximalCoupling(transport_coarse, transport_fine)

    coarse_state = ChainState(
        position=np.array([[0.0], [0.0]]), log_posterior=0.0, metadata={"iteration": 1}
    )
    fine_state = ChainState(
        position=np.array([[1.0], [1.0]]), log_posterior=0.0, metadata={"iteration": 1}
    )

    proposed_coarse, proposed_fine = coupling.sample_pair(coarse_state, fine_state)

    assert proposed_coarse.reference_position is not None
    assert proposed_fine.reference_position is not None


def test_proposal_base_update_parameters():
    proposal = ConcreteProposal(mu=np.zeros((2, 1)), cov=np.eye(2))
    new_mu = np.ones((2, 1))
    new_cov = 2.0 * np.eye(2)

    proposal.update_parameters(mu=new_mu, cov=new_cov)

    assert np.allclose(proposal.mu, new_mu)
    assert np.allclose(proposal.cov, new_cov)


def test_proposal_base_default_adapt_noop():
    proposal = ConcreteProposal(mu=np.zeros((2, 1)), cov=np.eye(2))
    state = ChainState(position=np.zeros((2, 1)), log_posterior=0.0, metadata={})
    proposal.adapt(state)
    assert np.allclose(proposal.mu, np.zeros((2, 1)))


def test_adapter_base_validates_window():
    with pytest.raises(ValueError):
        DummyAdapter(adapt_start=-1, adapt_end=10)

    with pytest.raises(ValueError):
        DummyAdapter(adapt_start=10, adapt_end=5)


def test_adaptive_proposal_delegates_sample_logpdf_and_attrs():
    base = ConcreteProposal(mu=np.zeros((2, 1)), cov=np.eye(2))
    adapter = ConcreteAdapter()
    wrapped = AdaptiveProposal(base, adapter)

    state = ChainState(
        position=np.array([[1.0], [2.0]]), log_posterior=-1.0, metadata={"iteration": 1}
    )
    proposed = wrapped.sample(state, common_step=np.ones((2, 1)))
    logq_fwd, logq_rev = wrapped.proposal_logpdf(state, proposed)

    assert np.allclose(proposed.position, np.array([[2.0], [3.0]]))
    assert logq_fwd == 0.0 and logq_rev == 0.0

    # __getattr__ delegation
    assert np.allclose(wrapped.mu, base.mu)

    # __setattr__ delegation
    wrapped.cov = 3.0 * np.eye(2)
    assert np.allclose(base.cov, 3.0 * np.eye(2))


def test_adaptive_proposal_calls_adapter():
    base = ConcreteProposal(mu=np.zeros((2, 1)), cov=np.eye(2))
    adapter = ConcreteAdapter()
    wrapped = AdaptiveProposal(base, adapter)
    state = ChainState(
        position=np.array([[3.0], [4.0]]), log_posterior=-1.0, metadata={"iteration": 1}
    )

    wrapped.adapt(state)

    assert adapter.calls == 1
    assert np.allclose(base.mu, state.position)


def test_adaptive_proposal_forwards_adapt_context():
    base = TrackingProposal(mu=np.zeros((2, 1)), cov=np.eye(2))
    adapter = ConcreteAdapter()
    wrapped = AdaptiveProposal(base, adapter)

    state = ChainState(
        position=np.array([[1.0], [1.0]]), log_posterior=-1.0, metadata={"iteration": 2}
    )
    samples = [state]
    paired_samples = [
        ChainState(
            position=np.array([[2.0], [2.0]]),
            log_posterior=-2.0,
            metadata={"iteration": 2},
        )
    ]

    wrapped.adapt(
        state,
        samples=samples,
        force_adapt=True,
        paired_samples=paired_samples,
    )

    assert len(base.adapt_calls) == 1
    call = base.adapt_calls[0]
    assert call["state"] is state
    assert call["samples"] is samples
    assert call["force_adapt"] is True
    assert call["paired_samples"] is paired_samples


def test_transport_proposal_adapt_forwards_context_to_proposal_and_map():
    base = TrackingProposal(mu=np.zeros((2, 1)), cov=np.eye(2))
    tmap = TrackingMap()
    wrapped = TransportProposalBase(base, tmap)

    state = ChainState(
        position=np.array([[0.0], [0.0]]), log_posterior=0.0, metadata={"iteration": 1}
    )
    samples = [state]
    paired_samples = [
        ChainState(
            position=np.array([[1.0], [1.0]]),
            log_posterior=-1.0,
            metadata={"iteration": 1},
        )
    ]

    wrapped.adapt(
        state,
        samples=samples,
        force_adapt=True,
        paired_samples=paired_samples,
    )

    assert len(base.adapt_calls) == 1
    proposal_call = base.adapt_calls[0]
    assert proposal_call["samples"] is samples
    assert proposal_call["force_adapt"] is True
    assert proposal_call["paired_samples"] is paired_samples

    assert len(tmap.adapt_calls) == 1
    map_call = tmap.adapt_calls[0]
    assert map_call["samples"] is samples
    assert map_call["force_adapt"] is True
    assert map_call["paired_samples"] is paired_samples


def test_coupled_proposal_adapt_pair_routes_samples_and_paired_samples():
    coarse_base = TrackingProposal(mu=np.zeros((2, 1)), cov=np.eye(2))
    fine_base = TrackingProposal(mu=np.zeros((2, 1)), cov=np.eye(2))
    coarse_map = TrackingMap()
    fine_map = TrackingMap()
    coarse_transport = TransportProposalBase(coarse_base, coarse_map)
    fine_transport = TransportProposalBase(fine_base, fine_map)
    coupled = InheritedCoupledProposal(coarse_transport, fine_transport)

    coarse_state = ChainState(
        position=np.array([[0.0], [0.0]]), log_posterior=0.0, metadata={"iteration": 3}
    )
    fine_state = ChainState(
        position=np.array([[1.0], [1.0]]), log_posterior=-1.0, metadata={"iteration": 3}
    )
    coarse_samples = [coarse_state]
    fine_samples = [fine_state]

    coupled.adapt_pair(
        coarse_state,
        fine_state,
        samples=(coarse_samples, fine_samples),
        force_adapt=True,
    )

    assert coarse_base.adapt_calls[0]["samples"] is coarse_samples
    assert coarse_base.adapt_calls[0]["paired_samples"] is fine_samples
    assert coarse_map.adapt_calls[0]["samples"] is coarse_samples
    assert coarse_map.adapt_calls[0]["paired_samples"] is fine_samples

    assert fine_base.adapt_calls[0]["samples"] is fine_samples
    assert fine_base.adapt_calls[0]["paired_samples"] is coarse_samples
    assert fine_map.adapt_calls[0]["samples"] is fine_samples
    assert fine_map.adapt_calls[0]["paired_samples"] is coarse_samples


def test_adaptive_proposal_satisfies_proposal_protocol():
    base = ConcreteProposal(mu=np.zeros((2, 1)), cov=np.eye(2))
    wrapped = AdaptiveProposal(base, ConcreteAdapter())
    assert isinstance(wrapped, ProposalBase)


def test_dummy_coupled_proposal_satisfies_coupled_protocol():
    base1 = ConcreteProposal(mu=np.zeros((2, 1)), cov=np.eye(2))
    base2 = ConcreteProposal(mu=np.zeros((2, 1)), cov=np.eye(2))
    coupled = DummyCoupledProposal(base1, base2)
    assert isinstance(coupled, CoupledProposalBase)
    assert coupled.proposal_coarse is base1
    assert coupled.proposal_fine is base2
