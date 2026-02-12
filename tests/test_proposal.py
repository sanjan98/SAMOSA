import numpy as np
import pytest

from samosa.core.proposal import AdapterBase, AdaptiveProposal, ProposalBase
from samosa.core.state import ChainState


class ConcreteProposal(ProposalBase):
    def sample(self, current_state: ChainState, common_step=None) -> ChainState:
        if common_step is None:
            step = np.zeros_like(current_state.position)
        else:
            step = common_step
        return ChainState(position=current_state.position + step)

    def proposal_logpdf(self, current_state: ChainState, proposed_state: ChainState) -> tuple[float, float]:
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

    state = ChainState(position=np.array([[1.0], [2.0]]), log_posterior=-1.0, metadata={"iteration": 1})
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
    state = ChainState(position=np.array([[3.0], [4.0]]), log_posterior=-1.0, metadata={"iteration": 1})

    wrapped.adapt(state)

    assert adapter.calls == 1
    assert np.allclose(base.mu, state.position)