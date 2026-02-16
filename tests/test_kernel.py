"""
Unit tests for samosa.core.kernel (_marginal_mh_acceptance_ratio and CoupledKernelBase).
"""

from typing import cast

import pytest
import numpy as np

from samosa.core.kernel import _marginal_mh_acceptance_ratio, CoupledKernelBase
from samosa.core.state import ChainState
from samosa.core.model import Model
from samosa.core.proposal import Proposal, CoupledProposal


# --------------------------------------------------
# _marginal_mh_acceptance_ratio
# --------------------------------------------------
def test_marginal_mh_acceptance_ratio_accept():
    """When proposed is better (higher log posterior + reverse), AR = 1."""
    ar = _marginal_mh_acceptance_ratio(
        current_log_posterior=-2.0,
        proposed_log_posterior=-0.5,
        logq_forward=0.0,
        logq_reverse=0.0,
    )
    assert ar == 1.0


def test_marginal_mh_acceptance_ratio_reject():
    """When proposed is worse, AR = exp(check)."""
    ar = _marginal_mh_acceptance_ratio(
        current_log_posterior=-0.5,
        proposed_log_posterior=-2.0,
        logq_forward=0.0,
        logq_reverse=0.0,
    )
    expected = np.exp((-2.0 + 0.0) - (-0.5 + 0.0))
    assert np.isclose(ar, expected)
    assert 0 <= ar < 1


def test_marginal_mh_acceptance_ratio_with_proposal_terms():
    """AR uses logq_forward and logq_reverse correctly."""
    # check = (proposed + logq_reverse) - (current + logq_forward)
    ar = _marginal_mh_acceptance_ratio(
        current_log_posterior=-1.0,
        proposed_log_posterior=-0.5,
        logq_forward=-0.2,
        logq_reverse=-0.1,
    )
    check = (-0.5 + (-0.1)) - (-1.0 + (-0.2))
    expected = 1.0 if check > 0 else float(np.exp(check))
    assert np.isclose(ar, expected)


# --------------------------------------------------
# CoupledKernelBase (mock)
# --------------------------------------------------
class MockCoupledProposal(CoupledProposal):
    """Coupled proposal that returns deterministic offsets for testing."""

    def __init__(self, proposal_coarse: Proposal, proposal_fine: Proposal) -> None:
        super().__init__(proposal_coarse, proposal_fine)

    def sample_pair(
        self, coarse_state: ChainState, fine_state: ChainState
    ) -> tuple[ChainState, ChainState]:
        # Return coarse_state + 0.1, fine_state + 0.1 as new positions
        offset = np.array([[0.1], [0.1]])
        pc = ChainState(
            position=coarse_state.position + offset,
            log_posterior=None,
            metadata=coarse_state.metadata,
        )
        pf = ChainState(
            position=fine_state.position + offset,
            log_posterior=None,
            metadata=fine_state.metadata,
        )
        return pc, pf


class MockModel:
    """Model returning log_posterior = -0.5 * sum(x^2)."""

    def __call__(self, params: np.ndarray) -> dict:
        return {"log_posterior": -0.5 * float(np.sum(params**2))}


class MockProposal(Proposal):
    """Minimal proposal for testing."""

    def __init__(self) -> None:
        super().__init__(mu=np.zeros((2, 1)), cov=np.eye(2))

    def sample(self, state: ChainState, eta=None) -> ChainState:
        return ChainState(
            position=state.position + 0.1, log_posterior=None, metadata=state.metadata
        )

    def proposal_logpdf(
        self, current: ChainState, proposed: ChainState
    ) -> tuple[float, float]:
        return 0.0, 0.0


class ConcreteCoupledKernel(CoupledKernelBase):
    """Concrete kernel that evaluates models on proposed positions."""

    def __init__(
        self, coarse_model: Model, fine_model: Model, coupled_proposal: CoupledProposal
    ) -> None:
        super().__init__(coarse_model, fine_model, coupled_proposal)

    def propose(
        self, current_coarse: ChainState, current_fine: ChainState
    ) -> tuple[ChainState, ChainState]:
        proposed_coarse, proposed_fine = self.coupled_proposal.sample_pair(
            current_coarse, current_fine
        )
        # Attach model outputs
        coarse_out = self.coarse_model(proposed_coarse.position)
        fine_out = self.fine_model(proposed_fine.position)
        pc = ChainState(
            position=proposed_coarse.position,
            **coarse_out,
            metadata=current_coarse.metadata or {},
        )
        pf = ChainState(
            position=proposed_fine.position,
            **fine_out,
            metadata=current_fine.metadata or {},
        )
        return pc, pf


@pytest.fixture
def coarse_model():
    return MockModel()


@pytest.fixture
def fine_model():
    return MockModel()


@pytest.fixture
def proposal_coarse():
    return MockProposal()


@pytest.fixture
def proposal_fine():
    return MockProposal()


@pytest.fixture
def coupled_proposal(proposal_coarse, proposal_fine):
    return MockCoupledProposal(proposal_coarse, proposal_fine)


@pytest.fixture
def coupled_kernel(coarse_model, fine_model, coupled_proposal):
    return ConcreteCoupledKernel(coarse_model, fine_model, coupled_proposal)


@pytest.fixture
def current_coarse():
    return ChainState(
        position=np.array([[0.0], [0.0]]),
        log_posterior=0.0,
        metadata={"iteration": 1},
    )


@pytest.fixture
def current_fine():
    return ChainState(
        position=np.array([[0.0], [0.0]]),
        log_posterior=0.0,
        metadata={"iteration": 1},
    )


def test_coupled_kernel_init_rejects_invalid_model():
    """CoupledKernelBase raises if coarse_model is not Model."""
    with pytest.raises(ValueError, match="coarse_model must be an instance of Model"):
        ConcreteCoupledKernel(
            cast(Model, None),
            MockModel(),
            MockCoupledProposal(MockProposal(), MockProposal()),
        )


def test_coupled_kernel_init_rejects_invalid_coupled_proposal(coarse_model, fine_model):
    """CoupledKernelBase raises if coupled_proposal is not CoupledProposal."""
    with pytest.raises(
        ValueError, match="coupled_proposal must be an instance of CoupledProposal"
    ):
        ConcreteCoupledKernel(coarse_model, fine_model, cast(CoupledProposal, None))


def test_coupled_kernel_propose(coupled_kernel, current_coarse, current_fine):
    """propose returns (proposed_coarse, proposed_fine) with model evaluations."""
    proposed_coarse, proposed_fine = coupled_kernel.propose(
        current_coarse, current_fine
    )
    assert proposed_coarse.position.shape == current_coarse.position.shape
    assert proposed_fine.position.shape == current_fine.position.shape
    assert proposed_coarse.log_posterior is not None
    assert proposed_fine.log_posterior is not None
    assert np.allclose(proposed_coarse.position, np.array([[0.1], [0.1]]))
    assert np.allclose(proposed_fine.position, np.array([[0.1], [0.1]]))


def test_coupled_kernel_acceptance_ratio(coupled_kernel, current_coarse, current_fine):
    """acceptance_ratio returns (ar_coarse, ar_fine) in [0, 1]."""
    proposed_coarse, proposed_fine = coupled_kernel.propose(
        current_coarse, current_fine
    )
    ar_c, ar_f = coupled_kernel.acceptance_ratio(
        current_coarse, proposed_coarse, current_fine, proposed_fine
    )
    assert 0 <= ar_c <= 1
    assert 0 <= ar_f <= 1


def test_coupled_kernel_adapt(coupled_kernel, current_coarse, current_fine):
    """adapt runs without error (proposals may not implement adapt)."""
    proposed_coarse, proposed_fine = coupled_kernel.propose(
        current_coarse, current_fine
    )
    coupled_kernel.adapt(proposed_coarse, proposed_fine)
