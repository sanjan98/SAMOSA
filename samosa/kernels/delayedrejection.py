"""
Delayed Rejection kernel for MCMC sampling.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from samosa.core.state import ChainState
from samosa.core.model import Model
from samosa.core.proposal import Proposal
from samosa.core.kernel import _marginal_mh_acceptance_ratio


class DelayedRejectionKernel:
    """
    Delayed Rejection kernel for MCMC sampling.
    Implements multiple proposal stages; each stage is tried only if
    all previous stages have been rejected.
    """

    def __init__(
        self, model: Model, proposal: Proposal, cov_scale: float = 0.5
    ) -> None:
        """
        Initialize the 2-stage Delayed Rejection kernel.

        Args:
            model: Model that computes log posterior given parameters.
            proposal: Proposal distribution.
            cov_scale: Scaling factor for covariance at second stage.
        """
        self.model = model
        if not isinstance(model, Model):
            raise ValueError("model must be an instance of Model")
        self.proposal = proposal
        if not isinstance(proposal, Proposal):
            raise ValueError("proposal must be an instance of Proposal")
        self.cov_scale = cov_scale
        self.first_stage_state: ChainState | None = None

    def propose(self, state: ChainState) -> ChainState:
        """
        Generate a candidate state from the current state (one or two stages).

        Args:
            state: Current state of the chain.

        Returns:
            Proposed state of the chain (or current state if both stages are rejected).
        """
        proposed_1_state = self._proposestate(state)
        self.first_stage_state = proposed_1_state
        ar1 = self.acceptance_ratio(state, proposed_1_state)

        u = np.random.rand()
        if ar1 == 1.0 or u < ar1:
            self.ar = ar1
            return proposed_1_state
        # First stage rejected: propose second stage with scaled covariance
        original_cov = self.proposal.cov.copy()
        self.proposal.cov = original_cov * self.cov_scale
        proposed_2_state = self._proposestate(state)
        self.proposal.cov = original_cov
        ar2 = self._second_stage_acceptance_ratio(
            state, proposed_1_state, proposed_2_state
        )
        self.ar = ar2
        u2 = np.random.rand()
        if ar2 == 1.0 or u2 < ar2:
            return proposed_2_state
        return state

    def acceptance_ratio(self, current: ChainState, proposed: ChainState) -> float:
        """Compute acceptance probability for the proposed state."""
        logq_forward, logq_reverse = self.proposal.proposal_logpdf(current, proposed)
        lpc = current.log_posterior if current.log_posterior is not None else -np.inf
        lpp = proposed.log_posterior if proposed.log_posterior is not None else -np.inf
        return _marginal_mh_acceptance_ratio(lpc, lpp, logq_forward, logq_reverse)

    def _second_stage_acceptance_ratio(
        self, current: ChainState, first_stage: ChainState, second_stage: ChainState
    ) -> float:
        """Compute delayed-rejection acceptance ratio for the second stage."""
        if second_stage.log_posterior is None or second_stage.log_posterior == -np.inf:
            return 0.0

        logq_forward_1, logq_reverse_1 = self.proposal.proposal_logpdf(
            current, first_stage
        )

        original_cov = self.proposal.cov.copy()
        self.proposal.cov = original_cov * self.cov_scale
        logq_forward_2, logq_reverse_2 = self.proposal.proposal_logpdf(
            current, second_stage
        )
        logq_y2_to_y1, logq_y1_to_y2 = self.proposal.proposal_logpdf(
            second_stage, first_stage
        )
        self.proposal.cov = original_cov

        lpc = current.log_posterior if current.log_posterior is not None else -np.inf
        lpf1 = (
            first_stage.log_posterior
            if first_stage.log_posterior is not None
            else -np.inf
        )
        lpf2 = (
            second_stage.log_posterior
            if second_stage.log_posterior is not None
            else -np.inf
        )

        # First-stage acceptance probs: alpha_1 from x, alpha_1_reverse = hypothetical from y_2
        check_alpha_1 = (lpf1 - lpc) + (logq_reverse_1 - logq_forward_1)
        alpha_1 = 1.0 if check_alpha_1 > 0 else float(np.exp(check_alpha_1))

        check_alpha_1_reverse = (lpf1 - lpf2) + (logq_y1_to_y2 - logq_y2_to_y1)
        alpha_1_reverse = (
            1.0 if check_alpha_1_reverse > 0 else float(np.exp(check_alpha_1_reverse))
        )

        # Green & Mira: alpha_2 = min(1, (1-alpha_1(y_2))/(1-alpha_1(y_1)) * pi(y_2)/pi(x) * q2(x|y_2)/q2(y_2|x) * q1(y_1|y_2)/q1(y_1|x))
        # log A = log(1-alpha_1_reverse) - log(1-alpha_1) + (lpf2-lpc) + (logq_reverse_2-logq_forward_2) + (logq_y2_to_y1 - logq_forward_1)
        denominator = 1.0 - alpha_1
        if denominator < 1e-10:
            return 0.0
        numerator = (1.0 - alpha_1_reverse) * np.exp(
            (lpf2 - lpc)
            + (logq_reverse_2 - logq_forward_2)
            + (logq_y2_to_y1 - logq_forward_1)
        )
        return min(1.0, numerator / denominator)

    def adapt(
        self,
        proposed: ChainState,
        *,
        samples: Optional[list[ChainState]] = None,
        force_adapt: Optional[bool] = False,
    ) -> None:
        """Adapt the proposal based on the proposed state.

        Args:
            proposed: Proposed state of the chain.
            samples: Optional list of samples from the chain.
            force_adapt: Optional force-adaptation flag.
        """
        self.proposal.adapt(proposed, samples=samples, force_adapt=force_adapt)

    def _proposestate(self, current_state: ChainState) -> ChainState:
        """Generate a candidate state from the current state using the proposal."""
        proposed = self.proposal.sample(current_state)
        model_output = self.model(proposed.position)
        return ChainState(
            position=proposed.position,
            reference_position=proposed.reference_position,
            **model_output,
            metadata=current_state.metadata.copy()
            if current_state.metadata is not None
            else None,
        )
