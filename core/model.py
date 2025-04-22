"""
Template class file for model
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class BaseModel(ABC):
    """
    Abstract base class for MCMC models.
    Supports both direct posterior and prior/likelihood workflows.
    """

    @abstractmethod
    def __call__(self, params: np.ndarray) -> Dict[str, Any]:
        """
        Compute probability components and model outputs.
        Returns a dict containing either:
            - 'log_posterior' (direct)
            - or both 'prior' and 'likelihood'
        May include: 'model_output', 'cost', 'qoi', etc.
        """
        pass