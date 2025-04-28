"""
Template class file for transport map
"""

# Imports
import numpy as np
from typing import Protocol, List, Tuple
from samosa.core.state import ChainState

class TransportMap(Protocol):
    """
    Protocol for Transport Maps
    """

    def forward(self, position: np.ndarray) -> Tuple[np.ndarray, float]:
        """Transport the position to the reference space"""
        raise NotImplementedError("Implement forward method")
    
    def inverse(self, position: np.ndarray) -> Tuple[np.ndarray, float]:
        """Transport the position back to the original space"""
        raise NotImplementedError("Implement inverse method")
    
    def adapt(self, samples: List['ChainState']) -> None:
        """Adapt the map based on history of samples"""
        raise NotImplementedError("Implement adapt method")