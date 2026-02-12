"""
Model interfaces for MCMC sampling.

This module provides the ModelProtocol that all models must implement.
Models are validated automatically when used with samplers.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ModelProtocol(Protocol):
    """
    Protocol defining the required interface for MCMC models.

    A model must be callable, accept a parameter vector of shape (d, 1),
    and return a dictionary with posterior information specified via:

    **Option 1 - Direct posterior:**
        Return dict with 'log_posterior' key.

    **Option 2 - Component-based:**
        Return dict with both 'log_prior' and 'log_likelihood' keys.

    **Optional keys** (custom metadata):
        'model_output', 'cost', 'qoi', 'gradient', etc.

    Examples:
        Function::

            def my_model(params: np.ndarray) -> dict:
                return {"log_posterior": -0.5 * np.sum(params ** 2)}

        Class::

            class MyModel:
                def __call__(self, params: np.ndarray) -> dict:
                    return {"log_posterior": -0.5 * np.sum(params ** 2)}

        Component-based::

            def my_model(params: np.ndarray) -> dict:
                return {
                    "log_prior": -0.5 * np.sum(params ** 2),
                    "log_likelihood": -np.sum((params - 1.0) ** 2),
                }
    """

    def __call__(self, params: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model at given parameters.

        Args:
            params: Parameter vector of shape (d, 1).

        Returns:
            Dictionary with posterior specification:
            - 'log_posterior' (float), OR
            - 'log_prior' (float) + 'log_likelihood' (float)
        """
        ...


def validate_model_output(output: Dict[str, Any]) -> None:
    """
    Validate that model output satisfies ModelProtocol.

    Checks that output contains proper posterior specification.
    Called automatically by samplers when model is evaluated.

    Args:
        output: Model output dictionary to validate.

    Raises:
        TypeError: If output is not a dict.
        ValueError: If posterior specification is invalid.
    """
    if not isinstance(output, dict):
        raise TypeError(f"Model must return dict, got {type(output).__name__}.")

    has_log_posterior = "log_posterior" in output
    has_log_prior = "log_prior" in output
    has_log_likelihood = "log_likelihood" in output

    # Must have either log_posterior or both components
    if not has_log_posterior and not (has_log_prior and has_log_likelihood):
        raise ValueError(
            f"Model output must contain either:\n"
            f"  1. 'log_posterior' key, OR\n"
            f"  2. Both 'log_prior' AND 'log_likelihood' keys\n"
            f"Got: {list(output.keys())}"
        )

    # Check for partial specifications
    if (has_log_prior and not has_log_likelihood) or (
        has_log_likelihood and not has_log_prior
    ):
        raise ValueError(
            f"If using component specification, must provide BOTH "
            f"'log_prior' and 'log_likelihood'. "
            f"Got: {list(output.keys())}"
        )

    # Check for mixed specifications
    if has_log_posterior and (has_log_prior or has_log_likelihood):
        raise ValueError(
            f"Cannot mix 'log_posterior' with component specifications. "
            f"Got: {list(output.keys())}"
        )
