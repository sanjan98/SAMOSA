"""
Class file for the lower triangular map using MParT
Currently the source distribution is assumed to be a standard Gaussian as otherwise we will need the gradient of the logpdf of the source distribution for the optimization
Maybe this feature can be added later (probably not)
"""

# Imports
import numpy as np
from scipy.linalg import sqrtm

from samosa.core.map import TransportMap
from samosa.core.state import ChainState
from samosa.core.model import ModelProtocol
from samosa.utils.post_processing import get_position_from_states
from typing import List, Optional

class LinearOptimalTransportMap(TransportMap):
    """
    Class for the lower triangular map using MParT. 
    """

    def __init__(self, dim: int, adapt_start: int = 500, adapt_end: int = 1000, adapt_interval: int = 100, reference_model: Optional[ModelProtocol] = None, eps: Optional[float] = 1e-6):
        
        """
        Initialize the optimal map
        Args:
            dim: Dimension of the map.
            adapt_start: Start iteration for adaptation.
            adapt_end: End iteration for adaptation.
            adapt_interval: Interval for adaptation.
            reference_model: Optional reference model for the map (If none is provided, a standard Gaussian is assumed).
        """

        self.dim = dim
        self.adapt_start = adapt_start
        self.adapt_end = adapt_end
        self.adapt_interval = adapt_interval
        self.reference_model = reference_model
        self.A = np.eye(dim)
        self.b = np.zeros((dim, 1))
        self.eps = eps

    def forward(self, x):
        """
        Forward map to transform the input x using the triangular map.
        
        Args: 
            x: Input data to be transformed.
        Returns:
            Transformed data and log determinant.
        """
        # Evaluate the map
        r = self.A @ x + self.b
        
        # Also return the log determinant
        log_det = np.log(np.abs(np.linalg.det(self.A)))
        
        # Return the transformed data and log determinant
        return r, log_det
    
    def inverse(self, r):
        """
        Inverse map to transform the input r back to the original space.
        Args:
            r: Input data to be transformed back.
        Returns:
            Transformed data and log determinant.
        """
        # Evaluate the inverse map
        x = np.linalg.inv(self.A) @ (r - self.b)
        
        log_det = np.log(np.abs(np.linalg.det(np.linalg.inv(self.A))))
        # Return the inverse transformed data and log determinant 
        return x, log_det
    
    def adapt(self, samples_coarse: List[ChainState], samples_fine: List[ChainState], force_adapt: bool = False):
        """
        Adapt the map to new samples.
        
        Args:
            samples: New samples to adapt the map to.
        """
        
        # Get current iteration
        iteration = samples_fine[-1].metadata['iteration'] + 1

        # Only check conditions if not forcing adaptation
        if not force_adapt:
            # Check adaptation window
            if iteration < self.adapt_start or iteration >= self.adapt_end:
                return None
            
            # Check adaptation interval
            if (iteration - self.adapt_start) % self.adapt_interval != 0:
                return None
        
        print(f"Adapting Linear Optimal Transport map at iteration {iteration}")

        # Get positions from states
        positions_coarse = get_position_from_states(samples_coarse)
        positions_fine = get_position_from_states(samples_fine)

        self.mu_fine = np.mean(positions_fine, axis=1, keepdims=True)
        self.cov_fine = np.cov(positions_fine) + self.eps * np.eye(self.dim)

        # Check if reference model is provided
        if self.reference_model is None:         
            self.mu_coarse = np.zeros((self.dim, 1))
            self.cov_coarse = np.eye(self.dim)
        else:
            self.mu_coarse = np.mean(positions_coarse, axis=1, keepdims=True)
            self.cov_coarse = np.cov(positions_coarse) + self.eps * np.eye(self.dim)
        
        # Compute the optimal linear transport map (Fine -> Coarse)
        # Using the formula for linear optimal transport between Gaussians
        sqrt_cov_coarse = sqrtm(self.cov_coarse)
        inv_sqrt = np.linalg.inv(sqrtm(sqrt_cov_coarse @ self.cov_fine @ sqrt_cov_coarse))
        self.A = sqrt_cov_coarse @ inv_sqrt @ sqrt_cov_coarse
        self.b = self.mu_coarse - self.A @ self.mu_fine