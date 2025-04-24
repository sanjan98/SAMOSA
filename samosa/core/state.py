"""
Class file for the chain.
"""
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np

@dataclass
class ChainState:
    """
    Chain state class holding the current position and posterior value among other attributes.
    
    Attributes:
        position (np.ndarray): Current position in the parameter space.
        log_posterior (Optional[float]): Log posterior value. If not provided, it will be computed.
        prior (Optional[float]): Prior value. If not provided, it will be computed.
        likelihood (Optional[float]): Likelihood value. If not provided, it will be computed.
        model_output (Optional[np.ndarray]): Model output corresponding to the current position.
        qoi (Optional[np.ndarray]): Quantity of interest derived from the model output.
        cost (Optional[float]): Cost associated with the current state.
        metadata (Optional[dict]): Additional metadata for the state.
    
    Methods:
        __post_init__: Validates the state and computes missing values.
        _from_posterior: Class method to create a state from a posterior value.
        _from_components: Class method to create a state from prior and likelihood values.
        posterior: Property to access the log-posterior value.
    """
    # Necessary attributes
    position: np.ndarray

    # Optional to handle having direct knowledge of posterior (eg: for testing purposes)
    log_posterior: Optional[float] = None  
    prior: Optional[float] = None         
    likelihood: Optional[float] = None    
    
    # Optional attributes for additional information
    model_output: Optional[np.ndarray] = None
    qoi: Optional[np.ndarray] = None
    cost: Optional[float] = None

    # Optional attributes for metadata like iteration number, acceptance rate, etc.
    metadata: Optional[dict] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-compute log_posterior if possible, but don't validate"""
        if self.log_posterior is None and self.prior is not None and self.likelihood is not None:
            self.log_posterior = self.prior + self.likelihood

    @property
    def posterior(self) -> float:
        """Accessor with deferred validation"""
        if self.log_posterior is not None:
            return self.log_posterior
        
        if self.prior is not None and self.likelihood is not None:
            return self.prior + self.likelihood
            
        raise ValueError(
            "Posterior undefined. Set either: "
            "1. log_posterior directly, or "
            "2. both prior and likelihood"
        )

    def validate(self) -> None:
        """Explicit validation check"""
        if self.log_posterior is None and (self.prior is None or self.likelihood is None):
            raise ValueError("Invalid state: Must provide either log_posterior or both prior+likelihood")