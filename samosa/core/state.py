"""
Chain state representation for MCMC sampling.

This module provides the ChainState dataclass which encapsulates all information
about a single state in a Markov chain, including position, posterior components,
model outputs, and metadata.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class ChainState:
    """
    Represents the state of a Markov chain at a single iteration.

    A chain state encapsulates the current position in parameter space along with
    associated posterior components (prior, likelihood, posterior), model outputs,
    and sampling metadata. The state supports both direct posterior specification
    and component-based specification (prior + likelihood).

    Attributes:
        position (np.ndarray):
            Current position in parameter space. Must have shape (d, 1).

        reference_position (Optional[np.ndarray]):
            Position in reference/transformed space (e.g., from a transport map).
            Shape same as position. Default: None.

        log_posterior (Optional[float]):
            Log posterior value. If None and both log_prior and log_likelihood are
            provided, computed automatically. Default: None.

        log_prior (Optional[float]):
            Log prior probability. Used with log_likelihood to compute posterior.
            Default: None.

        log_likelihood (Optional[float]):
            Log likelihood. Used with log_prior to compute posterior. Default: None.

        model_output (Optional[np.ndarray]):
            Output from forward model evaluation (e.g., QoI, predictions).
            Shape determined by model. Default: None.

        qoi (Optional[np.ndarray]):
            Quantity of interest derived from model output or observations.
            Shape (m,) or (m, 1). Default: None.

        cost (Optional[float]):
            Computational cost of evaluating at this position.
            Useful for multi-fidelity methods. Default: None.

        metadata (Optional[Dict[str, Any]]):
            Additional state information such as:
            - 'iteration': Iteration number
            - 'acceptance_probability': Log acceptance ratio
            - 'is_accepted': Whether state was accepted
            - 'mean': Adaptive proposal mean
            - 'covariance': Adaptive proposal covariance
            - 'lambda': Global scaling factor
            Default: empty dict (None allowed).

    Examples:
        Direct posterior specification:
        >>> state = ChainState(
        ...     position=np.array([[1.0], [2.0]]),
        ...     log_posterior=-5.2,
        ...     metadata={'iteration': 1}
        ... )

        Component-based specification:
        >>> state = ChainState(
        ...     position=np.array([[1.0], [2.0]]),
        ...     log_prior=-2.0,
        ...     log_likelihood=-3.2,
        ...     metadata={'iteration': 1}
        ... )
        >>> state.posterior  # Returns -5.2 (computed automatically)

        With reference position (e.g., from transport map):
        >>> state = ChainState(
        ...     position=np.array([[1.0], [2.0]]),
        ...     reference_position=np.array([[0.1], [0.2]]),
        ...     log_prior=-2.0,
        ...     log_likelihood=-3.2
        ... )
    """

    # ==================== Required Attributes ====================
    position: np.ndarray
    """Current position in parameter space (d, 1)."""

    # ==================== Optional Reference Space ====================
    reference_position: Optional[np.ndarray] = None
    """Position in reference/transformed space for transport maps."""

    # ==================== Posterior Components ====================
    log_posterior: Optional[float] = None
    """Log posterior value (computed from prior + likelihood if not provided)."""

    log_prior: Optional[float] = None
    """Log prior probability."""

    log_likelihood: Optional[float] = None
    """Log likelihood probability."""

    # ==================== Model Information ====================
    model_output: Optional[np.ndarray] = None
    """Output from forward model evaluation."""

    qoi: Optional[np.ndarray] = None
    """Quantity of interest (observations or derived outputs)."""

    cost: Optional[float] = None
    """Computational cost of evaluating at this position."""

    # ==================== Metadata ====================
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    """
    Dictionary for additional state information:
    - 'iteration': Chain iteration number
    - 'acceptance_probability': Log acceptance ratio
    - 'is_accepted': Whether state was accepted
    - 'mean': Adaptive proposal mean
    - 'covariance': Adaptive proposal covariance
    - 'lambda': Global scaling factor
    """

    def __post_init__(self) -> None:
        """
        Auto-compute log_posterior from components if not provided.

        This allows flexibility in specifying states either as:
        1. Direct posterior (set log_posterior), or
        2. Components (set log_prior + log_likelihood)

        This method is called automatically after initialization.
        """
        if not isinstance(self.position, np.ndarray):
            raise TypeError("position must be a numpy.ndarray with shape (d, 1).")
        if self.position.ndim != 2 or self.position.shape[1] != 1:
            raise ValueError(
                f"position must have shape (d, 1), got {self.position.shape}."
            )

        if self.reference_position is not None:
            if not isinstance(self.reference_position, np.ndarray):
                raise TypeError(
                    "reference_position must be a numpy.ndarray with shape (d, 1)."
                )
            if (
                self.reference_position.ndim != 2
                or self.reference_position.shape[1] != 1
            ):
                raise ValueError(
                    "reference_position must have shape (d, 1), "
                    f"got {self.reference_position.shape}."
                )
            if self.reference_position.shape != self.position.shape:
                raise ValueError(
                    "reference_position must have the same shape as position."
                )

        if (
            self.log_posterior is None
            and self.log_prior is not None
            and self.log_likelihood is not None
        ):
            self.log_posterior = self.log_prior + self.log_likelihood

    @property
    def posterior(self) -> float:
        """
        Get the log posterior value with deferred validation.

        Returns the log posterior if available, computing it from prior and
        likelihood if necessary. Only validates when called (lazy validation).

        Returns:
            float: Log posterior value.

        Raises:
            ValueError: If posterior cannot be determined (neither log_posterior
                nor both log_prior and log_likelihood are set).

        Examples:
            >>> state = ChainState(
            ...     position=np.array([[1.0]]),
            ...     log_prior=-1.0,
            ...     log_likelihood=-2.0
            ... )
            >>> state.posterior
            -3.0
        """
        if self.log_posterior is not None:
            return self.log_posterior

        if self.log_prior is not None and self.log_likelihood is not None:
            return self.log_prior + self.log_likelihood

        raise ValueError(
            "Posterior value cannot be determined. Must provide either:\n"
            "  1. log_posterior directly, or\n"
            "  2. both log_prior and log_likelihood"
        )

    def validate(self) -> None:
        """
        Explicitly validate that posterior can be computed.

        Use this method to validate state before sampling or when you want
        to catch errors early rather than deferring to property access.

        Raises:
            ValueError: If posterior cannot be computed from available data.

        Examples:
            >>> state = ChainState(position=np.array([[1.0]]))
            >>> state.validate()  # Raises ValueError

            >>> state = ChainState(
            ...     position=np.array([[1.0]]),
            ...     log_posterior=-5.0
            ... )
            >>> state.validate()  # OK
        """
        if self.log_posterior is None and (
            self.log_prior is None or self.log_likelihood is None
        ):
            raise ValueError(
                "Invalid state: Posterior cannot be determined. Must provide either:\n"
                "  1. log_posterior directly, or\n"
                "  2. both log_prior and log_likelihood"
            )

    def __repr__(self) -> str:
        """
        String representation of the state.

        Returns a concise representation showing position shape, posterior,
        and whether optional fields are present.

        Returns:
            str: Human-readable representation.
        """
        pos_shape = self.position.shape if self.position is not None else None
        posterior_str = (
            f"posterior={self.posterior:.4f}"
            if self.log_posterior is not None
            else "posterior=?"
        )

        optional_fields = []
        if self.reference_position is not None:
            optional_fields.append("ref_pos")
        if self.model_output is not None:
            optional_fields.append("model_output")
        if self.qoi is not None:
            optional_fields.append("qoi")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict or None.")
        if self.metadata:
            optional_fields.append(f"metadata({len(self.metadata)})")

        optional_str = f", {', '.join(optional_fields)}" if optional_fields else ""

        return f"ChainState(position_shape={pos_shape}, {posterior_str}{optional_str})"
