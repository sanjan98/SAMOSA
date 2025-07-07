"""
Class file for the lower triangular map using MParT
Currently the source distribution is assumed to be a standard Gaussian as otherwise we will need the gradient of the logpdf of the source distribution for the optimization
Maybe this feature can be added later (probably not)
"""

# Imports
import numpy as np
import mpart as mt

from samosa.core.map import TransportMap
from samosa.core.state import ChainState
from samosa.core.model import ModelProtocol
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from samosa.utils.post_processing import get_position_from_states
from samosa.utils.tools import lognormpdf
from typing import List, Optional

class LowerTriangularMap(TransportMap):
    """
    Class for the lower triangular map using MParT.
    """

    def __init__(self, dim: int, total_order: int = 2, adapt_start: int = 500, adapt_end: int = 1000, adapt_interval: int = 100, reference_model: Optional[ModelProtocol] = None, grad_reference_model: Optional[ModelProtocol] = None):
        
        """
        Initialize the lower triangular map.
        Args:
            dim: Dimension of the map.
            total_order: Total order of the map.ß
            adapt_start: Start iteration for adaptation.
            adapt_end: End iteration for adaptation.
            adapt_interval: Interval for adaptation.
            reference_model: Optional reference model for the map (If none is provided, a standard Gaussian is assumed).
        """

        self.dim = dim
        self.total_order = total_order
        self.adapt_start = adapt_start
        self.adapt_end = adapt_end
        self.adapt_interval = adapt_interval
        self.reference_model = reference_model
        self.grad_reference_model = grad_reference_model

        # Set some default values for mean and std
        self.mean = np.zeros((dim, 1))
        self.std = np.ones((dim, 1))

        # Define the map
        self._define_map()

    def forward(self, x):
        """
        Forward map to transform the input x using the triangular map.
        
        Args: 
            x: Input data to be transformed.
        Returns:
            Transformed data and log determinant.
        """
        # Scale the input first
        xscaled = (x - self.mean) / self.std
        # Evaluate the map
        r = self.ttm.Evaluate(xscaled)
        
        # Also return the log determinant
        log_det = self.ttm.LogDeterminant(xscaled) + np.log(np.prod(1 / self.std))
        
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
        xscaled = self.ttm.Inverse(r, r)
        # Rescale the output
        x = xscaled * self.std + self.mean
        
        log_det = -self.ttm.LogDeterminant(xscaled) + np.log(np.prod(self.std))
        # Return the inverse transformed data and log determinant 
        return x, log_det
    
    def adapt(self, samples: List[ChainState], force_adapt: bool = False):
        """
        Adapt the map to new samples.
        
        Args:
            samples: New samples to adapt the map to.
        """
        
        # Get current iteration
        iteration = samples[-1].metadata['iteration'] + 1

        # Only check conditions if not forcing adaptation
        if not force_adapt:
            # Check adaptation window
            if iteration < self.adapt_start or iteration >= self.adapt_end:
                return None
            
            # Check adaptation interval
            if (iteration - self.adapt_start) % self.adapt_interval != 0:
                return None
        
        print(f"Adapting LowerTriangular map at iteration {iteration}")

        # Get positions from states
        positions = get_position_from_states(samples)

        # Check if reference model is provided
        if self.reference_model is None:         
            # Standardize the positions of shape (dim, n_samples)
            self.mean = np.mean(positions, axis=1, keepdims=True)
            self.std = np.std(positions, axis=1, keepdims=True)
        else:
            self.mean = np.zeros((self.dim, 1))
            self.std = np.ones((self.dim, 1))
        
        # Standardize the positions
        x = (positions - self.mean) / self.std
        self.x = x

        # Optimize the map with the new samples
        self._optimize_map()

    def _define_map(self):
        """
        Define the lower triangular map using MParT.
        """
        # Prepare lists for components and their map options
        components = []
        for dim in range(1, self.dim + 1):
            # Create a FixedMultiIndexSet for each dimension
            fixed_mset = mt.FixedMultiIndexSet(dim, self.total_order)
            
            # Set up map options (customize if required)
            map_options = mt.MapOptions()
            # map_options.basisType = BasisTypes.HermiteFunctions
            
            # Create component from fixed multiindex set
            component = mt.CreateComponent(fixed_mset, map_options)
            components.append(component)

        # Create the triangular map using all components
        triangular_map = mt.TriangularMap(components)

        # Set coefficients for the map
        coefficients = np.concatenate([comp.CoeffMap() for comp in components])
        triangular_map.SetCoeffs(coefficients)

        self.ttm = triangular_map
        self.comps = components

    def _optimize_map(self):
        optimizer_options={'gtol': 1e-6, 'disp': True}

        if self.reference_model is None:

            # Loop through each component to print initial coefficients, compute objectives, and optimize
            for idx, (component, x_segment) in enumerate(zip(self.comps, [self.x[:i+1, :] for i in range(self.dim)])):
                print('==================')
                print(f'Starting coeffs component {idx + 1}:')
                print(component.CoeffMap())
                print(f'Objective value for component {idx + 1}: {self.obj(component.CoeffMap(), component, x_segment):.2E}')
                print('==================')

                # Optimize for each component
                res = minimize(self.obj, component.CoeffMap(), args=(component, x_segment), jac=self.grad_obj, method='BFGS', options=optimizer_options)

                # Print final coeffs and objective
                print('==================')
                print(f'Final coeffs component {idx + 1}:')
                print(component.CoeffMap())
                print(f'Objective value for component {idx + 1}: {self.obj(component.CoeffMap(), component, x_segment):.2E}')
                print('==================')
        
        else:
            # If a reference model is provided, we can use it to optimize the map

            # Print initial coefficients and objective
            print('==================')
            print('Starting coeffs (reference model case):')
            print(self.ttm.CoeffMap())
            print(f'Objective value (reference model case): {self.obj(self.ttm.CoeffMap(), self.ttm, self.x):.2E}')
            print('==================')

            if self.grad_reference_model is not None:
                res = minimize(self.obj, self.ttm.CoeffMap(), args=(self.ttm, self.x), jac=self.grad_obj, method='BFGS', options=optimizer_options)
            else:
                # If no gradient reference model is provided, use the default gradient computation
                res = minimize(self.obj, self.ttm.CoeffMap(), args=(self.ttm, self.x), method='BFGS', options=optimizer_options)

            # Print final coefficients and objective
            print('==================')
            print('Final coeffs (reference model case):')
            print(self.ttm.CoeffMap())
            print(f'Objective value (reference model case): {self.obj(self.ttm.CoeffMap(), self.ttm, self.x):.2E}')
            print('==================')

    def pullback(self, x):
        """
        Pull back the input x using the inverse map.
        
        Args:
            x: Input data to be pulled back.
        Returns:
            Pulled back data pdf
        """
        
        # Compute the forward map
        r, logdet = self.forward(x)

        if self.reference_model is None:
            log_pullback_pdf = lognormpdf(r, np.zeros((self.dim, 1)), np.eye(self.dim)) + logdet
        else:
            # Use the reference model to compute the log pdf
            log_pullback_pdf = self.reference_model(r)['log_posterior'] + logdet
        pull_back_pdf = np.exp(log_pullback_pdf)
        return pull_back_pdf
    
    def checkpoint_model(self, filepath: str):
        """
        Save only the lower triangular map for later plotting/analysis
        
        Args:
            filepath: Path to save the model (use .pkl extension)
        """
        import pickle
        
        filepath = filepath + '.pkl'
        
        model_data = {
            'map_coefficients': self.ttm.CoeffMap(),
            'component_coeffs': [comp.CoeffMap() for comp in self.comps],
            'mean': self.mean,
            'std': self.std,
            'dim': self.dim,
            'total_order': self.total_order
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load the lower triangular map from a checkpoint file.
        
        Args:
            filepath: Path to the checkpoint file (use .pkl extension)
        """
        import pickle

        filepath = filepath + '.pkl'
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Load basic parameters
        self.mean = model_data['mean']
        self.std = model_data['std']
        self.dim = model_data['dim']
        self.total_order = model_data['total_order']
        
        # Recreate the map structure if needed
        self._define_map()
        
        # Load coefficients
        self.ttm.SetCoeffs(model_data['map_coefficients'])
        
        # Load individual component coefficients
        for comp, coeffs in zip(self.comps, model_data['component_coeffs']):
            comp.SetCoeffs(coeffs)
        
        print(f"Model loaded from {filepath}")

    # Negative log likelihood objective
    def obj(self, coeffs, tri_map, x):
        """ 
        Evaluates the log-likelihood of the samples using the map-induced density. 
        
        *** An important note: The samples x are already standardized in the _optimize_map method, so we can stick to using the native Evaluate and LogDeterminant methods of the map. As the standardization layer is an affine map, it does not have an effect on the "optimization" of the map coefficients. ***
        """
        num_points = x.shape[1]
        tri_map.SetCoeffs(coeffs)

        # Compute the map-induced density at each point
        map_of_x = tri_map.Evaluate(x)
        log_det = tri_map.LogDeterminant(x)

        if self.reference_model is None:
            # Reference density
            rho1 = multivariate_normal(np.zeros(1),np.eye(1))
            rho_of_map_of_x = rho1.logpdf(map_of_x.T)
        else:
            # Use the reference model to compute the density
            rho_of_map_of_x = self.reference_model(map_of_x)['log_posterior']

        # Return the negative log-likelihood of the entire dataset
        return -np.sum(rho_of_map_of_x + log_det)/num_points

    def grad_obj(self, coeffs, tri_map, x):
        """ Returns the gradient of the log-likelihood objective wrt the map parameters. """
        num_points = x.shape[1]
        tri_map.SetCoeffs(coeffs)

        # Evaluate the map
        map_of_x = tri_map.Evaluate(x)

        if self.reference_model is None:
            # Now compute the inner product of the map jacobian (\nabla_w S) and the gradient (which is just -S(x) here)
            grad_rho_of_map_of_x = tri_map.CoeffGrad(x, -map_of_x)
        else:
            if self.grad_reference_model is not None:
                # Use the reference model to compute the gradient of the log pdf
                grad_log_posterior = self.grad_reference_model(map_of_x)
                grad_rho_of_map_of_x = tri_map.CoeffGrad(x, grad_log_posterior)
            else:
                grad_rho_of_map_of_x = tri_map.CoeffGrad(x, -map_of_x)
        
        # Get the gradient of the log determinant with respect to the map coefficients
        grad_log_det = tri_map.LogDeterminantCoeffGrad(x)

        return -np.sum(grad_rho_of_map_of_x + grad_log_det, 1)/num_points
