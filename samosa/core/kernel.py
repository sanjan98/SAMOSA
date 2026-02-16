"""
Base classes for MCMC transition kernels.

Coupled kernels implement CoupledKernelBase with marginal MH acceptance and adapt
shared in the base; subclasses provide coarse_model, fine_model, and coupled_proposal.
"""

from __future__ import annotations

from abc import ABC
from typing import Optional

import numpy as np

from samosa.core.state import ChainState
from samosa.core.proposal import CoupledProposal
from samosa.core.model import Model


def _marginal_mh_acceptance_ratio(
    current_log_posterior: float,
    proposed_log_posterior: float,
    logq_forward: float,
    logq_reverse: float,
) -> float:
    """Compute the marginal MH acceptance ratio for the proposed state."""
    check = (proposed_log_posterior + logq_reverse) - (
        current_log_posterior + logq_forward
    )
    return 1.0 if check > 0 else float(np.exp(check))


class CoupledKernelBase(ABC):
    """
    Abstract base class for coupled (two-chain) kernels with marginal MH acceptance.

    Subclasses implement only propose(); acceptance_ratio and adapt are shared
    (marginal MH per chain and adapting both proposals). Subclasses must set
    coarse_model and fine_model in __init__ for use in propose().
    """

    def __init__(
        self,
        coarse_model: Model,
        fine_model: Model,
        coupled_proposal: CoupledProposal,
    ) -> None:
        """
        Initialize the coupled kernel.

        Args:
            coarse_model: Model for the coarse chain.
            fine_model: Model for the fine chain.
            coupled_proposal: Coupled proposal for the two chains.
        """
        self.coarse_model = coarse_model
        if not isinstance(coarse_model, Model):
            raise ValueError("coarse_model must be an instance of Model")
        self.fine_model = fine_model
        if not isinstance(fine_model, Model):
            raise ValueError("fine_model must be an instance of Model")
        self.coupled_proposal = coupled_proposal
        if not isinstance(coupled_proposal, CoupledProposal):
            raise ValueError("coupled_proposal must be an instance of CoupledProposal")

    def propose(
        self,
        current_coarse: ChainState,
        current_fine: ChainState,
    ) -> tuple[ChainState, ChainState]:
        """
        Generate coupled candidate states for both chains.

        Args:
            current_coarse: Current coarse state.
            current_fine: Current fine state.

        Returns:
            Tuple of (proposed_coarse_state, proposed_fine_state).
        """
        proposed_coarse, proposed_fine = self.coupled_proposal.sample_pair(
            current_coarse, current_fine
        )
        return proposed_coarse, proposed_fine

    def acceptance_ratio(
        self,
        current_coarse: ChainState,
        proposed_coarse: ChainState,
        current_fine: ChainState,
        proposed_fine: ChainState,
    ) -> tuple[float, float]:
        """
        Compute marginal MH acceptance ratio for each chain.

        Args:
            current_coarse: Current coarse state.
            proposed_coarse: Proposed coarse state.
            current_fine: Current fine state.
            proposed_fine: Proposed fine state.

        Returns:
            Tuple of (coarse_acceptance_ratio, fine_acceptance_ratio).
        """
        (logq_c_fwd, logq_c_rev), (logq_f_fwd, logq_f_rev) = (
            self.coupled_proposal.proposal_logpdf_pair(
                current_coarse, proposed_coarse, current_fine, proposed_fine
            )
        )
        lpc = (
            current_coarse.log_posterior
            if current_coarse.log_posterior is not None
            else -np.inf
        )
        lppc = (
            proposed_coarse.log_posterior
            if proposed_coarse.log_posterior is not None
            else -np.inf
        )
        lpf = (
            current_fine.log_posterior
            if current_fine.log_posterior is not None
            else -np.inf
        )
        lppf = (
            proposed_fine.log_posterior
            if proposed_fine.log_posterior is not None
            else -np.inf
        )
        ar_coarse = _marginal_mh_acceptance_ratio(lpc, lppc, logq_c_fwd, logq_c_rev)
        ar_fine = _marginal_mh_acceptance_ratio(lpf, lppf, logq_f_fwd, logq_f_rev)
        return ar_coarse, ar_fine

    def adapt(
        self,
        proposed_coarse: ChainState,
        proposed_fine: ChainState,
        *,
        samples: Optional[tuple[list[ChainState], list[ChainState]]] = None,
        force_adapt: bool = False,
    ) -> None:
        """
        Adapt both proposals using the proposed states and the samples.

        Args:
            proposed_coarse: Proposed coarse state.
            proposed_fine: Proposed fine state.
            samples: Optional tuple of (coarse_samples, fine_samples).
            force_adapt: Whether to force adaptation.
        """
        self.coupled_proposal.adapt_pair(
            proposed_coarse, proposed_fine, samples=samples, force_adapt=force_adapt
        )
