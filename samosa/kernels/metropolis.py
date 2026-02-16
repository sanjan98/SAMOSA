"""
Class file for the Metropolis-Hastings kernel
"""

from __future__ import annotations
import numpy as np
from samosa.core.state import ChainState
from samosa.core.proposal import Proposal
from samosa.core.model import Model
from samosa.core.kernel import _marginal_mh_acceptance_ratio
from typing import Optional


class MetropolisHastingsKernel:
    """
    Metropolis-Hastings kernel for MCMC sampling.
    """

    def __init__(self, model: Model, proposal: Proposal) -> None:
        """
        Initialize the Metropolis-Hastings kernel with a model.
        """
        self.model = model
        if not isinstance(model, Model):
            raise ValueError("model must be an instance of Model")
        self.proposal = proposal
        if not isinstance(proposal, Proposal):
            raise ValueError("proposal must be an instance of Proposal")

    def propose(self, state: ChainState) -> ChainState:
        """
        Generate candidate state from current state. Returns proposed state.

        Args:
            state: Current state of the chain.

        Returns:
            Proposed state of the chain.
        """
        proposed = self.proposal.sample(state)
        model_result = self.model(proposed.position)
        proposed_state = ChainState(
            position=proposed.position,
            reference_position=proposed.reference_position,
            **model_result,
            metadata=state.metadata.copy() if state.metadata is not None else None,
        )
        return proposed_state

    def acceptance_ratio(self, current: ChainState, proposed: ChainState) -> float:
        """Compute acceptance probability for the proposed state.

        Args:
            current: Current state of the chain.
            proposed: Proposed state of the chain.

        Returns:
            Acceptance probability for the proposed state.
        """
        logq_forward, logq_reverse = self.proposal.proposal_logpdf(current, proposed)
        lpc = current.log_posterior if current.log_posterior is not None else -np.inf
        lppc = proposed.log_posterior if proposed.log_posterior is not None else -np.inf
        ar = _marginal_mh_acceptance_ratio(lpc, lppc, logq_forward, logq_reverse)
        return ar

    def adapt(
        self,
        current: ChainState,
        *,
        samples: Optional[list[ChainState]] = None,
        force_adapt: Optional[bool] = False,
    ) -> None:
        """Adapt the proposal based on the proposed state.

        Args:
            current: Current state of the chain.
            samples: Optional list of samples from the chain.
            force_adapt: Optional force-adaptation flag.
        """
        self.proposal.adapt(current, samples=samples, force_adapt=force_adapt)
