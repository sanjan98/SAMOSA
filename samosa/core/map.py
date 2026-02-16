"""
Transport map base classes for MCMC sampling.

This module provides the TransportMapBase class for transport-based MCMC,
where sampling is performed in a reference space and transformed to the
target distribution via a bijective map.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
from samosa.core.state import ChainState
from samosa.utils.tools import lognormpdf


class TransportMapBase(ABC):
    """
    Abstract base class for transport maps.

    A transport map T transforms between target space π(x) and reference
    space ρ(r) (typically N(0, I)). Maps must be bijective and provide
    forward/inverse transforms with Jacobian determinants for acceptance ratios.

    All maps must implement forward(), inverse(), and adapt().
    The base class provides adaptation window/interval utilities so subclasses
    can implement their own adaptation strategy and signatures.

    Attributes:
        dim (int): Dimension of the parameter space.
        adapt_start (int): Iteration to start adaptation.
        adapt_end (int): Iteration to end adaptation.
        adapt_interval (int): Frequency of adaptation (every N iterations).

    Examples:
        Triangular map with MParT::

            class LowerTriangularMap(TransportMapBase):
                def __init__(self, dim, total_order=2):
                    super().__init__(dim=dim, adapt_start=500, adapt_end=10000)
                    self.total_order = total_order
                    self._define_mpart_map()

                def forward(self, position):
                    self.validate_position(position)
                    r = self.ttm.Evaluate(position)
                    logdet = self.ttm.LogDeterminant(position)
                    return r, logdet

                def inverse(self, reference_position):
                    x = self.ttm.Inverse(reference_position, reference_position)
                    logdet = -self.ttm.LogDeterminant(x)
                    return x, logdet

                def adapt(self, samples, force_adapt=False):
                    if not self._should_adapt(samples, force_adapt=force_adapt):
                        return
                    positions = get_position_from_states(samples)
                    self._optimize_mpart(positions)
    """

    def __init__(
        self,
        dim: int,
        adapt_start: int = 500,
        adapt_end: int = 10000,
        adapt_interval: int = 100,
    ) -> None:
        """
        Initialize transport map with dimension and adaptation settings.

        Args:
            dim: Dimension of the parameter space.
            adapt_start: Start adaptation at this iteration.
            adapt_end: Stop adaptation after this iteration.
            adapt_interval: Adapt every N iterations within window.

        Raises:
            ValueError: If dim <= 0 or adaptation window is invalid.
        """
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        if adapt_start < 0 or adapt_end < adapt_start:
            raise ValueError(
                f"Invalid adaptation window: must have 0 <= adapt_start <= adapt_end, "
                f"got adapt_start={adapt_start}, adapt_end={adapt_end}"
            )
        if adapt_interval <= 0:
            raise ValueError(f"adapt_interval must be positive, got {adapt_interval}")

        self.dim = dim
        self.adapt_start = adapt_start
        self.adapt_end = adapt_end
        self.adapt_interval = adapt_interval
        # Optional affine standardization parameters in target space.
        self.norm_mean = np.zeros((dim, 1))
        self.norm_std = np.ones((dim, 1))

    @abstractmethod
    def forward(
        self,
        position: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Transform position to reference space.

        Args:
            position: Position in target space, shape (d, 1).

        Returns:
            tuple of (reference_position, log_determinant).
                - reference_position: shape (d, 1)
                - log_determinant: log |det(∇T(x))|
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward()."
        )

    @abstractmethod
    def inverse(
        self,
        reference_position: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Transform position back to target space.

        Args:
            reference_position: Position in reference space, shape (d, 1).

        Returns:
            tuple of (position, log_determinant).
                - position: shape (d, 1)
                - log_determinant: log |det(∇T^{-1}(r))|
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement inverse()."
        )

    @abstractmethod
    def adapt(
        self,
        samples: list[ChainState],
        force_adapt: Optional[bool] = False,
        paired_samples: Optional[list[ChainState]] = None,
    ) -> None:
        """
        Adapt the map using user-defined strategy.

        Subclasses should implement adaptation logic and can accept additional
        arguments as needed. Use `_should_adapt(...)` to apply standard
        adaptation-window checks.

        Args:
            samples: List of ChainState objects to inform adaptation.
            force_adapt: If True, bypass window checks in subclass logic.
            paired_samples: Optional list of paired ChainState objects for coupled adaptation.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement adapt().")

    def checkpoint_model(self, filepath: str) -> None:
        """
        Optional checkpoint hook for map persistence.

        Subclasses can override this to persist map state.

        Args:
            filepath: Path to save the checkpoint.
        """
        return None

    def pullback_logpdf(
        self,
        position: np.ndarray,
        *,
        reference_logpdf: Optional[Callable[[np.ndarray], np.ndarray | float]] = None,
    ) -> np.ndarray | float:
        """
        Evaluate the pullback log-density at target-space position(s).

        Computes:
            log pi_map(x) = log rho(T(x)) + log |det(∇T(x))|

        Args:
            position: Target-space position(s), shape (d, N) or (d, 1).
            reference_logpdf: Optional callable for log rho(r). If omitted,
                uses `self.reference_model` when available, else standard normal.

        Returns:
            Scalar (N=1) or array (N,) of pullback log-density values.
        """
        reference_position, log_det = self.forward(position)
        if reference_logpdf is None:
            log_rho = self._reference_logpdf(reference_position)
        else:
            log_rho = reference_logpdf(reference_position)
        return np.asarray(log_rho) + np.asarray(log_det)

    def _fit_standardization(
        self,
        points: np.ndarray,
        *,
        inplace: bool = True,
        eps: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit affine standardization parameters `(norm_mean, norm_std)` in target space.

        Args:
            points: Input points with shape (d, N).
            inplace: If True, write `self.norm_mean` and `self.norm_std`.
            eps: Lower bound for std entries to avoid divide-by-zero.

        Returns:
            Tuple `(norm_mean, norm_std)` with shapes (d, 1), (d, 1).
        """
        if points.ndim != 2 or points.shape[0] != self.dim:
            raise ValueError(
                f"points must have shape (d, N) with d={self.dim}, got {points.shape}."
            )

        norm_mean = np.mean(points, axis=1, keepdims=True)
        norm_std = np.std(points, axis=1, keepdims=True)
        norm_std = np.maximum(norm_std, eps)

        if inplace:
            self.norm_mean = norm_mean
            self.norm_std = norm_std

        return norm_mean, norm_std

    def _scale_points(
        self, points: np.ndarray, *, normalize: bool = True
    ) -> np.ndarray:
        """
        Convert points to scaled target coordinates.

        Args:
            points: Input points with shape (d, N).
            normalize: If True, apply standardization using `self.norm_mean` and `self.norm_std`.
                       If False, denoromalize points by applying the inverse transformation.

        Returns:
            Normalized or denormalized points.
        """
        if points.ndim != 2 or points.shape[0] != self.dim:
            raise ValueError(
                f"points must have shape (d, N) with d={self.dim}, got {points.shape}."
            )
        if normalize:
            return (points - self.norm_mean) / self.norm_std
        else:
            return points * self.norm_std + self.norm_mean

    def _reference_logpdf(self, reference_position: np.ndarray) -> np.ndarray | float:
        """
        Evaluate reference log-density at reference position(s).

        Uses:
        1. `self.reference_model` when available and compatible.
        2. Standard normal log-density otherwise.
        """
        reference_model = getattr(self, "reference_model", None)
        if reference_model is None:
            return lognormpdf(
                reference_position,
                np.zeros((self.dim, 1)),
                np.eye(self.dim),
            )

        if callable(reference_model):
            model_result = reference_model(reference_position)
            if isinstance(model_result, dict):
                if "log_posterior" in model_result:
                    posterior = model_result["log_posterior"]
                    return np.asarray(posterior)
                raise ValueError(
                    "reference_model callable returned dict without 'log_posterior'."
                )
            return np.asarray(model_result)

        if hasattr(reference_model, "log_prob"):
            log_prob = reference_model.log_prob(reference_position)
            # Handle torch-like outputs without importing torch.
            if hasattr(log_prob, "detach"):
                log_prob = log_prob.detach()
            if hasattr(log_prob, "cpu"):
                log_prob = log_prob.cpu()
            if hasattr(log_prob, "numpy"):
                log_prob = log_prob.numpy()
            return log_prob

        raise TypeError(
            "Unable to evaluate reference density. Provide `reference_logpdf` callable "
            "to pullback_logpdf(), or set `reference_model` as callable/log_prob compatible."
        )

    def _extract_iteration(self, samples: list[ChainState]) -> int:
        """
        Extract the current iteration from the last sample metadata.

        If metadata is missing or malformed, defaults to 0.
        """
        if not samples:
            return 0

        metadata = samples[-1].metadata
        if metadata is None:
            return 0

        iteration = metadata.get("iteration", 0)
        return int(iteration) if iteration is not None else 0

    def _should_adapt(
        self,
        samples: list[ChainState],
        force_adapt: bool = False,
        iteration: int | None = None,
    ) -> bool:
        """
        Check whether adaptation should run based on window/interval settings.

        Args:
            samples: list of chain states used to infer iteration if needed.
            force_adapt: If True, skip window/interval checks.
            iteration: Optional explicit iteration override.

        Returns:
            True when adaptation should be performed, otherwise False.
        """
        if not samples:
            return False

        if not force_adapt:
            if iteration is None:
                iteration = self._extract_iteration(samples)
            if iteration < self.adapt_start or iteration >= self.adapt_end:
                return False
            if (iteration - self.adapt_start) % self.adapt_interval != 0:
                return False

        return True

    def __repr__(self) -> str:
        """String representation showing map configuration."""
        return (
            f"{self.__class__.__name__}(dim={self.dim}, "
            f"adapt_start={self.adapt_start}, adapt_end={self.adapt_end}, "
            f"adapt_interval={self.adapt_interval})"
        )


# Compatibility alias
TransportMap = TransportMapBase
