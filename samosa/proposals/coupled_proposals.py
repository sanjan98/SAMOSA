"""
Coupled proposal implementations. Used in coupled MCMC algorithms primarily for multi-fidelity inference.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import replace
from samosa.core.proposal import Proposal, CoupledProposal
from samosa.core.state import ChainState
from samosa.proposals.gaussianproposal import GaussianRandomWalk, IndependentProposal
from samosa.utils.tools import sample_multivariate_gaussian


def _unwrap_base_proposal(proposal: Proposal) -> Proposal:
    """
    Return the innermost proposal by unwrapping known wrapper structure.

    Adaptive/transport wrappers expose their wrapped proposal via `.proposal`.
    """
    base = proposal
    while hasattr(base, "proposal"):
        base = getattr(base, "proposal")
    return base


def _uses_transport(proposal: Optional[Proposal]) -> bool:
    """
    Return True if proposal stack contains a transport-aware wrapper.

    Transport wrappers expose a `.map` attribute.
    """
    current = proposal
    while current is not None:
        if hasattr(current, "map"):
            return True
        if hasattr(current, "proposal"):
            current = getattr(current, "proposal")
        else:
            break
    return False


def _state_for_proposal(proposal: Proposal, state: ChainState) -> ChainState:
    """
    Build a state representation compatible with proposal_logpdf.

    For transport proposals, reference_position must match that proposal's map.
    """
    if hasattr(proposal, "map"):
        reference_position, _ = getattr(proposal, "map").forward(state.position)
        return replace(state, reference_position=reference_position)
    return state


class SynceCoupling(CoupledProposal):
    """
    A common random number coupling. Simple, effective and fast to compute.
    Muchandimath, Sanjan, and Alex Gorodetsky.
    "Synchronized step Multilevel Markov chain Monte Carlo."
    arXiv preprint arXiv:2501.16538 (2025).
    """

    def __init__(self, proposal_coarse: Proposal, proposal_fine: Proposal) -> None:
        super().__init__(proposal_coarse, proposal_fine)
        coarse_base = _unwrap_base_proposal(proposal_coarse)
        fine_base = _unwrap_base_proposal(proposal_fine)
        if isinstance(coarse_base, IndependentProposal) or isinstance(
            fine_base, IndependentProposal
        ):
            raise ValueError(
                "SynceCoupling requires state-dependent base proposals. "
                "IndependentProposal is not allowed."
            )

    def sample_pair(
        self, coarse_state: ChainState, fine_state: ChainState
    ) -> Tuple[ChainState, ChainState]:

        dim = coarse_state.position.shape[0]
        assert dim == fine_state.position.shape[0], (
            "The dimensions of the two chains must be the same."
        )

        eta = sample_multivariate_gaussian(np.zeros((dim, 1)), np.eye(dim))
        proposed_coarse = self.proposal_coarse.sample(coarse_state, eta)
        proposed_fine = self.proposal_fine.sample(fine_state, eta)

        return proposed_coarse, proposed_fine


class IndependentCoupling(CoupledProposal):
    """
    Independent coupling. Propose a common sample from an independent sampler
    Madrigal-Cianci, Juan P., Fabio Nobile, and Raul Tempone.
    "Analysis of a class of multilevel Markov chain Monte Carlo algorithms
    based on independent Metropolis-Hastings."
    SIAM/ASA Journal on Uncertainty Quantification 11, no. 1 (2023): 91-138.
    """

    def __init__(
        self,
        proposal_coarse: Proposal,
        proposal_fine: Proposal,
        common_sampler: Optional[Proposal] = None,
    ) -> None:
        super().__init__(proposal_coarse, proposal_fine)
        coarse_base = _unwrap_base_proposal(proposal_coarse)
        fine_base = _unwrap_base_proposal(proposal_fine)
        if isinstance(coarse_base, GaussianRandomWalk) or isinstance(
            fine_base, GaussianRandomWalk
        ):
            raise ValueError(
                "IndendentCoupling does not allow GaussianRandomWalk as base proposal."
            )
        self.common_sampler = common_sampler
        common_sampler_base = (
            _unwrap_base_proposal(common_sampler)
            if common_sampler is not None
            else None
        )
        if common_sampler_base is not None and not isinstance(
            common_sampler_base, IndependentProposal
        ):
            raise ValueError("common_sampler must be an IndependentProposal or None.")
        uses_transport = _uses_transport(proposal_coarse) or _uses_transport(
            proposal_fine
        )
        if (
            uses_transport
            and common_sampler is not None
            and _uses_transport(common_sampler)
        ):
            raise ValueError(
                "For transport proposals, common_sampler must be in reference "
                "space (use IndependentProposal, not TransportProposal)."
            )

    def sample_pair(
        self, coarse_state: ChainState, fine_state: ChainState
    ) -> Tuple[ChainState, ChainState]:

        dim = coarse_state.position.shape[0]
        assert dim == fine_state.position.shape[0], (
            "The dimensions of the two chains must be the same."
        )

        if self.common_sampler is None:
            assert (
                coarse_state.metadata is not None and fine_state.metadata is not None
            ), "Metadata must be available to compute common sampler."
            common_mean = (
                coarse_state.metadata["mean"] + fine_state.metadata["mean"]
            ) / 2
            common_cov = (
                coarse_state.metadata["covariance"] + fine_state.metadata["covariance"]
            ) / 2
            self.common_sampler = IndependentProposal(mu=common_mean, cov=common_cov)

        eta = self.common_sampler.sample(coarse_state).position
        proposed_coarse = self.proposal_coarse.sample(coarse_state, eta)
        proposed_fine = self.proposal_fine.sample(fine_state, eta)

        return proposed_coarse, proposed_fine


class MaximalCoupling(CoupledProposal):
    """Try to force two chains to sample the same point using maximal coupling"""

    def __init__(self, proposal_coarse: Proposal, proposal_fine: Proposal) -> None:
        super().__init__(proposal_coarse, proposal_fine)

    def sample_pair(
        self, coarse_state: ChainState, fine_state: ChainState
    ) -> Tuple[ChainState, ChainState]:

        dim = coarse_state.position.shape[0]
        assert dim == fine_state.position.shape[0], (
            "The dimensions of the two chains must be the same."
        )

        x_fine = self.proposal_fine.sample(fine_state)
        fine_current = _state_for_proposal(self.proposal_fine, fine_state)
        coarse_current = _state_for_proposal(self.proposal_coarse, coarse_state)

        x_fine_for_fine = _state_for_proposal(self.proposal_fine, x_fine)
        x_fine_for_coarse = _state_for_proposal(self.proposal_coarse, x_fine)

        px = np.exp(
            self.proposal_fine.proposal_logpdf(fine_current, x_fine_for_fine)[0]
        )
        qx = np.exp(
            self.proposal_coarse.proposal_logpdf(coarse_current, x_fine_for_coarse)[0]
        )
        w = np.random.uniform(0, px)
        if w <= qx:
            # Return order is (coarse, fine)
            return x_fine_for_coarse, x_fine_for_fine
        else:
            while True:
                y_coarse = self.proposal_coarse.sample(coarse_state)
                y_for_coarse = _state_for_proposal(self.proposal_coarse, y_coarse)
                y_for_fine = _state_for_proposal(self.proposal_fine, y_coarse)

                py = np.exp(
                    self.proposal_fine.proposal_logpdf(fine_current, y_for_fine)[0]
                )
                qy = np.exp(
                    self.proposal_coarse.proposal_logpdf(coarse_current, y_for_coarse)[
                        0
                    ]
                )
                wstar = np.random.uniform(0, qy)
                if wstar >= py:
                    return y_for_coarse, x_fine_for_fine
