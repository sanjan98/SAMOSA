"""
Example implementation for sampling from a banana-shaped distribution
using the provided MCMC framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import os

from core.model import ModelProtocol
from core.state import ChainState
from proposals.gaussianproposal import GaussianRandomWalk
from proposals.adapters import HaarioAdapter, GlobalAdapter
from core.proposal import AdaptiveProposal
from kernels.metropolis import MetropolisHastingsKernel
from kernels.delayedrejection import DelayedRejectionKernel
from samplers.single_chain import MCMCsampler

import mcmc.utils.mcmc_utils_and_plot as utils


class BananaDistribution(ModelProtocol):
    """
    Implements a banana-shaped distribution in a 2D space.
    
    The banana shape is created by defining a distribution where:
    - The first dimension (x) follows a normal distribution.
    - The second dimension (y) is related to x^2, plus some noise.
    
    Parameters:
        a: Controls the curvature of the banana shape.
        b: Controls the variance in the y direction.
    """
    
    def __init__(self, a: float = 1.0, b: float = 1.0):
        self.a = a  # Controls the curvature of the banana
        self.b = b  # Controls the variance in the y direction
    
    def __call__(self, params: np.ndarray) -> Dict[str, Any]:
        """
        Compute log probability density of the banana distribution.
        
        Args:
            params: 2D array of parameters [x, y]
            
        Returns:
            Dictionary with log posterior value
        """
        x, y = params
        
        # Transform y to straighten out the banana
        y_transformed = y - self.b * (x**2 - self.a)
        
        # Compute log probability in the transformed space (standard normal)
        log_prob = -0.5 * (x**2 + y_transformed**2)
        
        return {
            'log_posterior': log_prob,
            'prior': 0.0,  # Not explicitly separating prior and likelihood
            'likelihood': log_prob
        }


def plot_samples(samples, output_dir: str = '.', title: str = 'Banana Distribution Samples'):
    """
    Plot the samples from the MCMC chain.
    
    Args:
        samples: List of ChainState objects
        output_dir: Directory to save the plot
        title: Title for the plot
    """
    # Extract positions from samples
    positions = np.array([s.position for s in samples])
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the trace (positions over time)
    plt.subplot(2, 2, 1)
    plt.plot(positions[:, 0], label='x')
    plt.title('Trace Plot (x)')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    
    plt.subplot(2, 2, 2)
    plt.plot(positions[:, 1], label='y')
    plt.title('Trace Plot (y)')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    
    # Plot the samples in the 2D space
    plt.subplot(2, 1, 2)
    plt.scatter(positions[:, 0], positions[:, 1], alpha=0.5, s=3)
    plt.title('Samples from Banana Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Add contour lines of the true density if possible
    try:
        # Create a grid of points
        x = np.linspace(min(positions[:, 0])-1, max(positions[:, 0])+1, 100)
        y = np.linspace(min(positions[:, 1])-1, max(positions[:, 1])+1, 100)
        X, Y = np.meshgrid(x, y)
        
        # Compute log density on the grid
        Z = np.zeros_like(X)
        model = BananaDistribution(a=1.0, b=1.0)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = model(np.array([X[i, j], Y[i, j]]))['log_posterior']
        
        # Plot contour lines
        plt.contour(X, Y, Z, levels=10, colors='k', alpha=0.3)
    except:
        pass  # Skip contour if it fails
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'banana_samples.png'))
    plt.close()


def run_banana_example(output_dir: str = '.', n_samples: int = 10000):
    """
    Run the banana distribution sampling example.
    
    Args:
        output_dir: Directory to save results
        n_samples: Number of samples to generate
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model and parameters
    model = BananaDistribution(a=1.0, b=1.0)
    
    # Starting point (away from the mode to test convergence)
    initial_position = np.array([[2.0], [2.0]])
    dim = initial_position.shape[0]
    
    # Setup proposal
    proposal_scale = 1.0
    base_proposal = GaussianRandomWalk(
        mu=np.zeros((dim, 1)),
        sigma=proposal_scale * np.eye(dim)
    )
    
    # Add adaptation
    adapter = HaarioAdapter(
        scale=2.4**2/dim,  # Optimal scaling for Gaussian targets
        adapt_start=100,
        adapt_end=5000
    )
    # adapter = GlobalAdapter(ar = 0.44, adapt_start=100, adapt_end=5000)

    proposal = AdaptiveProposal(base_proposal, adapter)
    
    # Choose kernel (Metropolis-Hastings or Delayed Rejection)
    # kernel = MetropolisHastingsKernel(model)
    kernel = DelayedRejectionKernel(model, cov_scale=0.5)
    
    # Initialize and run sampler
    sampler = MCMCsampler(
        model=model,
        kernel=kernel,
        proposal=proposal,
        initial_position=initial_position,
        n_iterations=n_samples
    )
    
    # Run the sampler
    sampler.run(output_dir)
    
    # Load samples for plotting
    import pickle
    with open(f"{output_dir}/samples.pkl", "rb") as f:
        samples = pickle.load(f)
    
    # Plot the results
    plot_samples(samples, output_dir)
    
    # Print acceptance rate
    acceptance_rate = sum(1 for i in range(1, len(samples)) 
                         if not np.array_equal(samples[i].position, samples[i-1].position)) / (len(samples)-1)
    print(f"Acceptance rate: {acceptance_rate:.2f}")
    
    # Return the samples for further analysis
    return samples


if __name__ == "__main__":
    # Run the example
    samples = run_banana_example(output_dir="./output", n_samples=10000)
    print(f"Generated {len(samples)} samples from the banana distribution")
    print("Results saved in ./output directory")