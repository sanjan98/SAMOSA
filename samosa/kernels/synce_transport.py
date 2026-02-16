"""
Transport SYNCE coupled kernel for two-chain MCMC with transport maps.

Proposes in reference space (with optional resync); maps back to target space.
Overrides acceptance_ratio to include log-determinant correction from the maps.
"""

import numpy as np
from typing import Any, List, Tuple

from dataclasses import replace

from samosa.core.state import ChainState
from samosa.core.model import ModelProtocol
from samosa.core.proposal import ProposalBase
from samosa.core.kernel import CoupledKernelBase
from samosa.utils.tools import sample_multivariate_gaussian


class TransportSYNCEKernel(CoupledKernelBase):
    """
    Coupled kernel with transport maps (coarse and fine) and optional resync.

    Proposes in reference space; inverse map gives target-space proposals.
    Acceptance ratio includes log-det correction (overridden here). Adapt and
    propose logic are transport-specific; adapt_maps and save_maps remain.
    """

    def __init__(
        self,
        coarse_model: ModelProtocol,
        fine_model: ModelProtocol,
        coarse_map: Any,
        fine_map: Any,
        w: float = 0.0,
    ) -> None:
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.coarse_map = coarse_map
        self.fine_map = fine_map
        self.w = w
        if self.fine_map.reference_model is None:
            self.coupletype = "direct"
        else:
            self.coupletype = "deep"

    def propose(
        self,
        proposal_coarse: ProposalBase,
        proposal_fine: ProposalBase,
        current_coarse_state: ChainState,
        current_fine_state: ChainState,
    ) -> Tuple[ChainState, ChainState, ChainState, ChainState]:
        dim = current_coarse_state.position.shape[0]
        assert dim == current_fine_state.position.shape[0], (
            "The dimensions of the two chains must be the same."
        )

        coarse_theta = current_coarse_state.position
        fine_theta = current_fine_state.position

        coarse_r, logdet_current_coarse = self.coarse_map.forward(coarse_theta)
        current_coarse_state = replace(
            current_coarse_state, reference_position=coarse_r
        )
        current_coarse_state.metadata["logdetT"] = logdet_current_coarse

        if self.coupletype == "deep":
            ftoc_theta, logdet_ftoc_fine = self.fine_map.forward(fine_theta)
            fine_r, logdet_ctor_fine = self.coarse_map.forward(ftoc_theta)
            logdet_current_fine = logdet_ftoc_fine + logdet_ctor_fine
        else:
            fine_r, logdet_current_fine = self.fine_map.forward(fine_theta)

        current_fine_state = replace(current_fine_state, reference_position=fine_r)
        current_fine_state.metadata["logdetT"] = logdet_current_fine

        eta = sample_multivariate_gaussian(np.zeros((dim, 1)), np.eye(dim))
        u = np.random.rand()

        def get_cov(proposal: ProposalBase) -> np.ndarray:
            if hasattr(proposal, "cov"):
                return proposal.cov
            return proposal.proposal.cov

        if u < self.w:
            coarse_rprime = np.linalg.cholesky(get_cov(proposal_coarse)) @ eta
            fine_rprime = np.linalg.cholesky(get_cov(proposal_fine)) @ eta
        else:
            coarse_rprime = (
                coarse_r + np.linalg.cholesky(get_cov(proposal_coarse)) @ eta
            )
            fine_rprime = fine_r + np.linalg.cholesky(get_cov(proposal_fine)) @ eta

        coarse_thetaprime, logdet_proposed_coarse = self.coarse_map.inverse(
            coarse_rprime
        )

        if self.coupletype == "deep":
            rtoc_tetaprime, logdet_rtoc_fine = self.coarse_map.inverse(fine_rprime)
            fine_thetaprime, logdet_ctof_fine = self.fine_map.inverse(rtoc_tetaprime)
            logdet_proposed_fine = logdet_rtoc_fine + logdet_ctof_fine
        else:
            fine_thetaprime, logdet_proposed_fine = self.fine_map.inverse(fine_rprime)

        coarse_model_result = self.coarse_model(coarse_thetaprime)
        fine_model_result = self.fine_model(fine_thetaprime)

        proposed_coarse_state = ChainState(
            position=coarse_thetaprime,
            reference_position=coarse_rprime,
            **coarse_model_result,
            metadata=current_coarse_state.metadata.copy(),
        )
        proposed_coarse_state.metadata["logdetT"] = -logdet_proposed_coarse

        proposed_fine_state = ChainState(
            position=fine_thetaprime,
            reference_position=fine_rprime,
            **fine_model_result,
            metadata=current_fine_state.metadata.copy(),
        )
        proposed_fine_state.metadata["logdetT"] = -logdet_proposed_fine

        return (
            proposed_coarse_state,
            proposed_fine_state,
            current_coarse_state,
            current_fine_state,
        )

    def acceptance_ratio(
        self,
        proposal_coarse: ProposalBase,
        current_coarse: ChainState,
        proposed_coarse: ChainState,
        proposal_fine: ProposalBase,
        current_fine: ChainState,
        proposed_fine: ChainState,
    ) -> Tuple[float, float]:
        """Acceptance ratio with transport log-determinant correction."""
        coarse_r = current_coarse.reference_position
        coarse_rprime = proposed_coarse.reference_position
        logdet_current_coarse = current_coarse.metadata["logdetT"]
        logdet_proposed_coarse = proposed_coarse.metadata["logdetT"]

        current_reference_coarse = ChainState(position=coarse_r, log_posterior=None)
        proposed_reference_coarse = ChainState(
            position=coarse_rprime, log_posterior=None
        )
        logq_forward_coarse, logq_reverse_coarse = proposal_coarse.proposal_logpdf(
            current_reference_coarse, proposed_reference_coarse
        )
        check_coarse = (
            proposed_coarse.log_posterior + logq_reverse_coarse - logdet_proposed_coarse
        ) - (current_coarse.log_posterior + logq_forward_coarse - logdet_current_coarse)
        ar_coarse = 1.0 if check_coarse > 0 else float(np.exp(check_coarse))

        fine_r = current_fine.reference_position
        fine_rprime = proposed_fine.reference_position
        logdet_current_fine = current_fine.metadata["logdetT"]
        logdet_proposed_fine = proposed_fine.metadata["logdetT"]
        current_reference_fine = ChainState(position=fine_r, log_posterior=None)
        proposed_reference_fine = ChainState(position=fine_rprime, log_posterior=None)
        logq_forward_fine, logq_reverse_fine = proposal_fine.proposal_logpdf(
            current_reference_fine, proposed_reference_fine
        )
        check_fine = (
            proposed_fine.log_posterior + logq_reverse_fine - logdet_proposed_fine
        ) - (current_fine.log_posterior + logq_forward_fine - logdet_current_fine)
        ar_fine = 1.0 if check_fine > 0 else float(np.exp(check_fine))

        return ar_coarse, ar_fine

    def adapt_maps(
        self,
        samples_coarse: List[ChainState],
        samples_fine: List[ChainState],
    ) -> None:
        """Adapt the transport maps from coarse and fine samples."""
        self.coarse_map.adapt(samples_coarse)
        self.fine_map.adapt(samples_fine)

    def save_maps(self, output_dir: str, iteration: int) -> None:
        """Save transport map checkpoints."""
        if hasattr(self.coarse_map, "checkpoint_model"):
            self.coarse_map.checkpoint_model(f"{output_dir}/coarse_map_{iteration}")
        if hasattr(self.fine_map, "checkpoint_model"):
            self.fine_map.checkpoint_model(f"{output_dir}/fine_map_{iteration}")
