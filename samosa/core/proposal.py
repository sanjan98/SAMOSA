"""
Base classes for MCMC proposal distributions and adaptation strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Any
from samosa.core.state import ChainState
from samosa.core.map import TransportMap


class ProposalBase(ABC):
    """
    Abstract base class for MCMC proposal distributions.

    All proposals must implement sample() and proposal_logpdf() methods.
    Proposals can optionally implement adapt() for self-adaptation.

    Attributes:
        mu (np.ndarray): Mean vector of the proposal distribution.
        cov (np.ndarray): Covariance matrix of the proposal distribution.
    """

    def __init__(self, mu: np.ndarray, cov: np.ndarray) -> None:
        """
        Initialize proposal with mean and covariance.

        Args:
            mu: Mean vector (d, 1).
            cov: Covariance matrix (d, d).
        """
        self.mu = mu
        self.cov = cov

    @abstractmethod
    def sample(
        self, current_state: ChainState, common_step: Optional[np.ndarray] = None
    ) -> ChainState:
        """
        Generate a candidate state from the proposal distribution.

        Args:
            current_state: Current state of the chain.
            common_step: Optional common random variable for coupling.

        Returns:
            Proposed ChainState.
        """
        raise NotImplementedError("Subclasses must implement sample()")

    @abstractmethod
    def proposal_logpdf(
        self, current_state: ChainState, proposed_state: ChainState
    ) -> tuple[float, float]:
        """
        Compute forward and reverse log probabilities.

        Args:
            current_state: Current state of the chain.
            proposed_state: Proposed state.

        Returns:
            Tuple of (logq_forward, logq_reverse) where:
                - logq_forward: log q(proposed | current)
                - logq_reverse: log q(current | proposed)
        """
        raise NotImplementedError("Subclasses must implement proposal_logpdf()")

    def update_parameters(
        self, mu: Optional[np.ndarray] = None, cov: Optional[np.ndarray] = None
    ) -> None:
        """
        Update proposal parameters (called by adapters).

        Args:
            mu: New mean vector (optional).
            cov: New covariance matrix (optional).
        """
        if mu is not None:
            self.mu = mu
        if cov is not None:
            self.cov = cov

    def adapt(
        self,
        state: ChainState,
        *,
        samples: Optional[list[ChainState]] = None,
        force_adapt: Optional[bool] = False,
        paired_samples: Optional[list[ChainState]] = None,
    ) -> None:
        """
        Self-adapt proposal based on current state (optional).

        Override this method if the proposal has its own adaptation logic
        (e.g., adaptive beta in pCN proposals).

        Args:
            state: Current state containing metadata for adaptation.
            samples: Optional history for adaptation.
            force_adapt: Optional force-adaptation flag.
            paired_samples: Optional history from a coupled chain.
        """
        pass


class AdapterBase(ABC):
    """
    Abstract base class for proposal adaptation strategies.

    Adapters modify proposal parameters based on chain history to
    improve sampling efficiency (e.g., Haario covariance adaptation).

    Attributes:
        adapt_start: Iteration to start adaptation.
        adapt_end: Iteration to end adaptation.
        eps: Regularization parameter for covariance updates.
    """

    def __init__(
        self, adapt_start: int = 500, adapt_end: int = 1000, eps: float = 1e-06
    ) -> None:
        """
        Initialize adapter with adaptation window and regularization.

        Args:
            adapt_start: Start adaptation at this iteration.
            adapt_end: Stop adaptation after this iteration.
            eps: Small constant for numerical stability.
        """
        if adapt_start < 0 or adapt_end < adapt_start:
            raise ValueError(
                "Invalid adaptation window: must have 0 <= adapt_start <= adapt_end"
            )

        self.adapt_start = adapt_start
        self.adapt_end = adapt_end
        self.eps = eps

    @abstractmethod
    def adapt(self, proposal: ProposalBase, state: ChainState) -> None:
        """
        Adapt proposal parameters based on current state.

        Args:
            proposal: The proposal distribution to adapt.
            state: Current state containing adaptation metadata.
        """
        raise NotImplementedError("Subclasses must implement adapt()")


class AdaptiveProposal(ProposalBase):
    """
    Wrapper that applies an adapter to a base proposal.

    Delegates all proposal operations to the wrapped base proposal,
    but uses the adapter's strategy for parameter updates.

    Attributes:
        proposal: The wrapped base proposal.
        adapter: The adaptation strategy.
    """

    def __init__(self, base_proposal: ProposalBase, adapter: AdapterBase) -> None:
        """
        Wrap a proposal with an adapter.

        Args:
            base_proposal: The proposal to wrap.
            adapter: The adaptation strategy to use.
        """
        self.proposal = base_proposal
        self.adapter = adapter

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped proposal."""
        return getattr(self.proposal, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate attribute setting to wrapped proposal, except wrapper attributes."""
        if name in ("proposal", "adapter") or "proposal" not in self.__dict__:
            super().__setattr__(name, value)
        else:
            setattr(self.proposal, name, value)

    def adapt(
        self,
        state: ChainState,
        *,
        samples: Optional[list[ChainState]] = None,
        force_adapt: bool = False,
        paired_samples: Optional[list[ChainState]] = None,
    ) -> None:
        """Use the adapter to adapt the wrapped proposal."""
        self.adapter.adapt(self.proposal, state)
        self.proposal.adapt(
            state,
            samples=samples,
            force_adapt=force_adapt,
            paired_samples=paired_samples,
        )

    # Explicit delegation keeps this wrapper structurally compatible with protocols.
    def sample(
        self, current_state: ChainState, common_step: Optional[np.ndarray] = None
    ) -> ChainState:
        return self.proposal.sample(current_state, common_step)

    def proposal_logpdf(
        self, current_state: ChainState, proposed_state: ChainState
    ) -> tuple[float, float]:
        return self.proposal.proposal_logpdf(current_state, proposed_state)


class TransportProposalBase(ProposalBase):
    """
    Base class for single-chain transport-aware proposals.

    A transport proposal wraps a base proposal operating in reference space
    and a transport map for moving between target and reference spaces.

    Attributes:
        proposal: Wrapped base proposal used in reference space.
        map: Transport map with forward/inverse transforms.
    """

    def __init__(self, proposal: ProposalBase, map: TransportMap) -> None:
        """
        Initialize transport-aware proposal wrapper.

        Args:
            proposal: Base proposal operating in reference coordinates.
            map: Transport map used for forward/inverse transforms.
        """
        self.proposal = proposal
        self.map = map
        self._cache = {}

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to wrapped proposal."""
        return getattr(self.proposal, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate attribute setting to wrapped proposal, except wrapper attributes."""
        if name in ("proposal", "map", "_cache") or "proposal" not in self.__dict__:
            super().__setattr__(name, value)
        else:
            setattr(self.proposal, name, value)

    def sample(
        self, current_state: ChainState, common_step: Optional[np.ndarray] = None
    ) -> ChainState:
        """
        Generate a candidate state using the transport-aware proposal.
        1. Map current state to reference space: r = T(x)
        2. Store log determinant for proposal correction: log |det(∇T^{-1}(r))| = -log |det(∇T(x))|
        3. Sample in reference space using the wrapped proposal: r' ~ q(r' | r)
        4. Map proposed reference state back to target space: x' = T^{-1}(r')
        5. Store log determinant for reverse correction: log |det(∇T^{-1}(r'))|

        Args:
            current_state: Current state of the chain.
            common_step: Optional common random variable for coupling.

        Returns:
            Proposed ChainState.
        """
        reference_position, log_det_forward = self.map.forward(current_state.position)
        proposed_reference_state = self.proposal.sample(
            ChainState(position=reference_position, metadata=current_state.metadata),
            common_step,
        )
        proposed_position, log_det_reverse = self.map.inverse(
            proposed_reference_state.position
        )
        self._cache["current_position"] = np.array(current_state.position, copy=True)
        self._cache["proposed_position"] = np.array(proposed_position, copy=True)
        self._cache["log_det_Tinv_current"] = -log_det_forward
        self._cache["log_det_Tinv_proposed"] = log_det_reverse
        return ChainState(
            position=proposed_position,
            reference_position=proposed_reference_state.position,
            metadata=current_state.metadata.copy() if current_state.metadata else {},
        )

    def proposal_logpdf(
        self, current_state: ChainState, proposed_state: ChainState
    ) -> tuple[float, float]:
        """
        Compute forward and reverse log probabilities with transport correction.

        Args:
            current_state: Current state of the chain.
            proposed_state: Proposed state.

        Returns:
            Tuple of (logq_forward, logq_reverse) where:
                - logq_forward: log q(proposed | current) = log q_r(r' | r) + log |det(∇T(x))|
                - logq_reverse: log q(current | proposed) = log q(r | r') + log |det(∇T^{-1}(r'))|
        """
        if (
            current_state.reference_position is None
            or proposed_state.reference_position is None
        ):
            raise ValueError(
                "reference_position must be set before calling proposal_logpdf"
            )

        logq_forward, logq_reverse = self.proposal.proposal_logpdf(
            ChainState(
                position=current_state.reference_position,
                metadata=current_state.metadata,
            ),
            ChainState(
                position=proposed_state.reference_position,
                metadata=proposed_state.metadata,
            ),
        )

        use_cache = (
            {
                "current_position",
                "proposed_position",
                "log_det_Tinv_current",
                "log_det_Tinv_proposed",
            }.issubset(self._cache)
            and np.array_equal(self._cache["current_position"], current_state.position)
            and np.array_equal(
                self._cache["proposed_position"], proposed_state.position
            )
        )
        if use_cache:
            log_det_tinv_current = self._cache["log_det_Tinv_current"]
            log_det_tinv_proposed = self._cache["log_det_Tinv_proposed"]
        else:
            _, log_det_forward_current = self.map.forward(current_state.position)
            _, log_det_forward_proposed = self.map.forward(proposed_state.position)
            log_det_tinv_current = -log_det_forward_current
            log_det_tinv_proposed = -log_det_forward_proposed

        return (
            logq_forward + log_det_tinv_current,
            logq_reverse + log_det_tinv_proposed,
        )

    def adapt(
        self,
        state: ChainState,
        *,
        samples: Optional[list[ChainState]] = None,
        force_adapt: bool = False,
        paired_samples: Optional[list[ChainState]] = None,
    ) -> None:
        """
        Adapt wrapped proposal and transport map.

        Args:
            state: Current chain state.
            samples: Optional list of chain states for map adaptation.
                     If not provided, uses `[state]`.
            force_adapt: Bool to force adaptation.
            paired_samples: Optional history from coupled chain. Used for
                maps that require both chains during adaptation.
        """
        self.proposal.adapt(
            state,
            samples=samples,
            force_adapt=force_adapt,
            paired_samples=paired_samples,
        )

        history = samples if samples is not None else [state]
        self.map.adapt(
            history,
            force_adapt=force_adapt,
            paired_samples=paired_samples,
        )

    def save_map(self, output_dir: str, iteration: int) -> None:
        """
        Optional map checkpoint hook.

        Delegates to map.checkpoint_model when available.
        """
        if hasattr(self.map, "checkpoint_model"):
            self.map.checkpoint_model(f"{output_dir}/map_{iteration}")


class CoupledProposalBase(ABC):
    """
    Abstract base class for coupled proposals.

    A coupled proposal is composed of two single-chain proposals
    (coarse and fine) and generates correlated candidates for both chains.
    This lets kernels stay generic while coupling logic lives in proposals.

    Attributes:
        proposal_coarse: Coarse chain proposal. Can be any ProposalBase.
        proposal_fine: Fine chain proposal. Can be any ProposalBase.
    """

    def __init__(
        self,
        proposal_coarse: ProposalBase,
        proposal_fine: ProposalBase,
    ) -> None:
        """
        Initialize coupled proposal from two single-chain proposals.

        Args:
            proposal_coarse: Proposal for the coarse chain.
            proposal_fine: Proposal for the fine chain.
        """
        self.proposal_coarse = proposal_coarse
        self.proposal_fine = proposal_fine

    @abstractmethod
    def sample_pair(
        self,
        coarse_state: ChainState,
        fine_state: ChainState,
    ) -> tuple[ChainState, ChainState]:
        """
        Generate coupled candidate states for coarse and fine chains.

        Args:
            coarse_state: Current state of the coarse chain.
            fine_state: Current state of the fine chain.

        Returns:
            Tuple of (proposed_coarse_state, proposed_fine_state).
        """
        raise NotImplementedError("Subclasses must implement sample_pair()")

    def proposal_logpdf_pair(
        self,
        current_coarse: ChainState,
        proposed_coarse: ChainState,
        current_fine: ChainState,
        proposed_fine: ChainState,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Compute per-chain forward/reverse proposal log densities.

        Args:
            current_coarse: Current coarse state.
            proposed_coarse: Proposed coarse state.
            current_fine: Current fine state.
            proposed_fine: Proposed fine state.

        Returns:
            ((coarse_logq_forward, coarse_logq_reverse),
             (fine_logq_forward, fine_logq_reverse)).
        """
        coarse_logq_forward, coarse_logq_reverse = self.proposal_coarse.proposal_logpdf(
            current_coarse, proposed_coarse
        )
        fine_logq_forward, fine_logq_reverse = self.proposal_fine.proposal_logpdf(
            current_fine, proposed_fine
        )
        return (coarse_logq_forward, coarse_logq_reverse), (
            fine_logq_forward,
            fine_logq_reverse,
        )

    def adapt_pair(
        self,
        coarse_state: ChainState,
        fine_state: ChainState,
        *,
        samples: Optional[tuple[list[ChainState], list[ChainState]]] = None,
        force_adapt: bool = False,
    ) -> None:
        """
        Optional adaptation hook for coupled proposals.

        Delegates adaptation to the two wrapped proposals.

        Args:
            coarse_state: Current coarse state.
            fine_state: Current fine state.
            samples: Optional tuple of (coarse_samples, fine_samples).
                Passed through to proposals when provided.
            force_adapt: Passed through to proposals.
        """
        coarse_samples = samples[0] if samples is not None else None
        fine_samples = samples[1] if samples is not None else None

        self.proposal_coarse.adapt(
            coarse_state,
            samples=coarse_samples,
            force_adapt=force_adapt,
            paired_samples=fine_samples,
        )
        self.proposal_fine.adapt(
            fine_state,
            samples=fine_samples,
            force_adapt=force_adapt,
            paired_samples=coarse_samples,
        )


# Compatibility alias used in several kernels/samplers.
Proposal = ProposalBase
CoupledProposal = CoupledProposalBase
TransportProposal = TransportProposalBase
