"""
Coupled proposal implementations. Used in coupled MCMC algorithms primarily for multi-fidelity inference.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import replace
from samosa.core.proposal import Proposal, CoupledProposal
from samosa.core.state import ChainState
from samosa.proposals.gaussianproposal import IndependentProposal
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


def _state_for_proposal(proposal: Proposal, state: ChainState) -> ChainState:
    """
    Build a state representation compatible with proposal_logpdf.

    For transport proposals, reference_position must match that proposal's map.
    """
    if hasattr(proposal, "map"):
        reference_position, _ = getattr(proposal, "map").forward(state.position)
        return replace(state, reference_position=reference_position)
    return state


def _extract_mu_cov(base_proposal: Proposal) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract Gaussian parameters from a base proposal.

    IndependentCoupling's common sampler only needs `mu` and `cov`, so allow any
    proposal exposing these attributes (not just specific proposal subclasses).
    """
    if not hasattr(base_proposal, "mu") or not hasattr(base_proposal, "cov"):
        raise ValueError(
            "IndependentCoupling requires base proposals with `mu` and `cov` "
            "attributes to build a common independent sampler."
        )
    mu = np.asarray(getattr(base_proposal, "mu"))
    cov = np.asarray(getattr(base_proposal, "cov"))
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Base proposal covariance `cov` must be square.")
    if mu.shape[0] != cov.shape[0]:
        raise ValueError(
            "Base proposal mean/covariance shape mismatch while building common sampler."
        )
    return mu, cov


class SynceCoupling(CoupledProposal):
    def __init__(
        self, proposal_coarse: Proposal, proposal_fine: Proposal, omega: float = 0.0
    ) -> None:
        """
        A common random number coupling. Simple, effective and fast to compute.
        Muchandimath, Sanjan, and Alex Gorodetsky.
        "Synchronized step Multilevel Markov chain Monte Carlo."
        arXiv preprint arXiv:2501.16538 (2025).
        """
        super().__init__(proposal_coarse, proposal_fine)
        if not (0.0 <= omega <= 1.0):
            raise ValueError("omega must be in [0, 1].")
        self.omega = float(omega)
        self._last_mode = "synce"
        self.common_sampler: Optional[IndependentProposal] = None
        coarse_base = _unwrap_base_proposal(proposal_coarse)
        fine_base = _unwrap_base_proposal(proposal_fine)
        if isinstance(coarse_base, IndependentProposal) or isinstance(
            fine_base, IndependentProposal
        ):
            raise ValueError(
                "SynceCoupling requires state-dependent base proposals. "
                "IndependentProposal is not allowed."
            )
        # Lazily constructed only if omega-branch is used.
        self._independent_coupling: Optional[IndependentCoupling] = None

    def sample_pair(
        self, coarse_state: ChainState, fine_state: ChainState
    ) -> Tuple[ChainState, ChainState]:

        dim = coarse_state.position.shape[0]
        assert dim == fine_state.position.shape[0], (
            "The dimensions of the two chains must be the same."
        )

        # With probability omega, do an independent "resync" move.
        if np.random.rand() < self.omega:
            self._last_mode = "independent"
            if self._independent_coupling is None:
                self._independent_coupling = IndependentCoupling(
                    self.proposal_coarse, self.proposal_fine
                )
            proposed_coarse, proposed_fine = self._independent_coupling.sample_pair(
                coarse_state, fine_state
            )
            # Expose the current common sampler for diagnostics/backward compatibility.
            self.common_sampler = self._independent_coupling.common_sampler
        else:
            self._last_mode = "synce"
            eta = sample_multivariate_gaussian(np.zeros((dim, 1)), np.eye(dim))
            proposed_coarse = self.proposal_coarse.sample(coarse_state, eta)
            proposed_fine = self.proposal_fine.sample(fine_state, eta)

        return proposed_coarse, proposed_fine

    def proposal_logpdf_pair(
        self,
        current_coarse: ChainState,
        proposed_coarse: ChainState,
        current_fine: ChainState,
        proposed_fine: ChainState,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if self._last_mode == "independent":
            if self._independent_coupling is None:
                raise RuntimeError(
                    "SynceCoupling independent branch used before initialization."
                )
            return self._independent_coupling.proposal_logpdf_pair(
                current_coarse, proposed_coarse, current_fine, proposed_fine
            )
        return super().proposal_logpdf_pair(
            current_coarse, proposed_coarse, current_fine, proposed_fine
        )


class IndependentCoupling(CoupledProposal):
    def __init__(self, proposal_coarse: Proposal, proposal_fine: Proposal) -> None:
        """
        Independent coupling. Propose a common sample from an independent sampler
        Madrigal-Cianci, Juan P., Fabio Nobile, and Raul Tempone.
        "Analysis of a class of multilevel Markov chain Monte Carlo algorithms
        based on independent Metropolis-Hastings."
        SIAM/ASA Journal on Uncertainty Quantification 11, no. 1 (2023): 91-138.

        A common independent sampler is constructed from the current coarse/fine
        proposal parameters and refreshed in `sample_pair` so adaptation is reflected.
        Both chains then use the same shared reference sample.
        """
        super().__init__(proposal_coarse, proposal_fine)
        self.common_sampler = self._build_common_sampler_from_proposals()

    def _build_common_sampler_from_proposals(self) -> IndependentProposal:
        """
        Build common independent proposal from coarse/fine base proposals.

        Accepts GaussianRandomWalk or IndependentProposal bases (including wrappers).
        """
        coarse_base = _unwrap_base_proposal(self.proposal_coarse)
        fine_base = _unwrap_base_proposal(self.proposal_fine)
        coarse_mu, coarse_cov = _extract_mu_cov(coarse_base)
        fine_mu, fine_cov = _extract_mu_cov(fine_base)
        common_mean = 0.5 * (coarse_mu + fine_mu)
        common_cov = 0.5 * (coarse_cov + fine_cov)
        return IndependentProposal(mu=common_mean, cov=common_cov)

    def _state_from_common_reference(
        self, source_proposal: Proposal, reference_position: np.ndarray
    ) -> ChainState:
        """Map shared reference sample to proposal target-space state."""
        if hasattr(source_proposal, "map"):
            position, _ = getattr(source_proposal, "map").inverse(reference_position)
            return ChainState(
                position=position,
                reference_position=np.array(reference_position, copy=True),
            )
        return ChainState(position=np.array(reference_position, copy=True))

    def _proposal_logpdf_from_common_sampler(
        self,
        current: ChainState,
        proposed: ChainState,
        proposal: Proposal,
    ) -> tuple[float, float]:
        """Compute induced marginal logpdf under the common sampler.

        Reuses TransportProposal.get_log_det_tinv when available (cache + non-finite
        guard). Avoids calling map.forward on non-finite positions to prevent hangs.
        """
        if hasattr(proposal, "map"):
            map_obj = getattr(proposal, "map")
            current_ref = current.reference_position
            proposed_ref = proposed.reference_position
            # Do not call map.forward on non-finite positions (e.g. from failed inverse).
            if current_ref is None:
                if not np.all(np.isfinite(current.position)):
                    return (0.0, float("-inf"))
                current_ref, _ = map_obj.forward(current.position)
            if proposed_ref is None:
                if not np.all(np.isfinite(proposed.position)):
                    return (0.0, float("-inf"))
                proposed_ref, _ = map_obj.forward(proposed.position)

            if not np.all(np.isfinite(current_ref)) or not np.all(
                np.isfinite(proposed_ref)
            ):
                return (0.0, float("-inf"))

            logq_forward_ref, logq_reverse_ref = self.common_sampler.proposal_logpdf(
                ChainState(position=current_ref, metadata=current.metadata),
                ChainState(position=proposed_ref, metadata=proposed.metadata),
            )
            # Reuse TransportProposal cache and non-finite guard when available.
            get_log_det_tinv = getattr(proposal, "get_log_det_tinv", None)
            if get_log_det_tinv is not None:
                log_det_tinv_current, log_det_tinv_proposed = get_log_det_tinv(
                    current, proposed
                )
            else:
                if not np.all(np.isfinite(current.position)) or not np.all(
                    np.isfinite(proposed.position)
                ):
                    log_det_tinv_current = -np.inf
                    log_det_tinv_proposed = -np.inf
                else:
                    _, log_det_fwd_cur = map_obj.forward(current.position)
                    _, log_det_fwd_prop = map_obj.forward(proposed.position)
                    log_det_tinv_current = -log_det_fwd_cur
                    log_det_tinv_proposed = -log_det_fwd_prop

            logq_forward = logq_forward_ref + log_det_tinv_current
            logq_reverse = logq_reverse_ref + log_det_tinv_proposed
            if not np.isfinite(logq_forward) or not np.isfinite(logq_reverse):
                return (0.0, float("-inf"))
            return (logq_forward, logq_reverse)
        return self.common_sampler.proposal_logpdf(current, proposed)

    def sample_pair(
        self, coarse_state: ChainState, fine_state: ChainState
    ) -> Tuple[ChainState, ChainState]:

        dim = coarse_state.position.shape[0]
        assert dim == fine_state.position.shape[0], (
            "The dimensions of the two chains must be the same."
        )

        # Refresh from proposals so adaptation propagates to common sampler.
        self.common_sampler = self._build_common_sampler_from_proposals()
        common_reference = self.common_sampler.sample(coarse_state).position
        proposed_coarse = self._state_from_common_reference(
            self.proposal_coarse, common_reference
        )
        proposed_fine = self._state_from_common_reference(
            self.proposal_fine, common_reference
        )

        return proposed_coarse, proposed_fine

    def proposal_logpdf_pair(
        self,
        current_coarse: ChainState,
        proposed_coarse: ChainState,
        current_fine: ChainState,
        proposed_fine: ChainState,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Use common sampler induced marginals for both chains."""
        coarse_logq_forward, coarse_logq_reverse = (
            self._proposal_logpdf_from_common_sampler(
                current_coarse, proposed_coarse, self.proposal_coarse
            )
        )
        fine_logq_forward, fine_logq_reverse = (
            self._proposal_logpdf_from_common_sampler(
                current_fine, proposed_fine, self.proposal_fine
            )
        )
        return (coarse_logq_forward, coarse_logq_reverse), (
            fine_logq_forward,
            fine_logq_reverse,
        )


class MaximalCoupling(CoupledProposal):
    def __init__(self, proposal_coarse: Proposal, proposal_fine: Proposal) -> None:
        """
        Thorisson, Hermann. "Coupling methods in probability theory." Scandinavian journal of statistics (1995): 159-182.
        Try to force two chains to sample the same point using maximal coupling.
        """
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
