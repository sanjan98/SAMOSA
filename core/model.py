"""
Template class file for model
"""

from typing import Dict, Any, Protocol
import numpy as np

class ModelProtocol(Protocol):
    """
    Protocol for MCMC models.
    Supports both direct posterior and prior/likelihood workflows.
    """

    def __call__(self, params: np.ndarray) -> Dict[str, Any]:
        """
        Compute probability components and model outputs.
        Returns a dict containing either:
            - 'log_posterior' (direct)
            - or both 'prior' and 'likelihood'
        May include: 'model_output', 'cost', 'qoi', etc.
        """
        pass