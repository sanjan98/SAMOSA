# Some imports
import numpy as np
import sys
import os
from samosa.utils.tools import log_banana
from typing import Any, Dict
import matplotlib.pyplot as plt
import pickle

# Lets define the banana model
# The inputs will be the sample (x: np.ndarray of shape (n_dim, n_samples))
# The outputs will be the log posterior (log_posterior: np.ndarray of shape (n_samples,)) or the log likelihood and the log prior
# In this case, we will define the log posterior directly
# Note that other additional quantities like qoi, cost_model_output etc. can be defined

def banana_model(x: np.ndarray) -> Dict[str, Any]:
    """
    Banana model function
    """
    output = {}
    # Just use the log_banana function to compute the log posterior
    log_posterior = log_banana(x)

    output['log_posterior'] = log_posterior

    # If you want to compute the log likelihood and log prior separately, you can do it like this
    # log_likelihood = <Some custom function>(x)
    # log_prior = <Some custom function>(x)
    # output['log_likelihood'] = log_likelihood
    # output['log_prior'] = log_prior

    # If you want to compute the qoi, cost_model_output etc. you can do it like this
    cost = 2
    qoi = np.sum(x, axis=0)

    output['cost'] = cost
    output['qoi'] = qoi

    return output

# Lets mix and match some kernels and proposals and run MCMC

# This is just a simple example of how to use the samosa package
from samosa.kernels.metropolis import MetropolisHastingsKernel
from samosa.kernels.delayedrejection import DelayedRejectionKernel
from samosa.proposals.gaussianproposal import GaussianRandomWalk, IndependentProposal
from samosa.proposals.adapters import HaarioAdapter, GlobalAdapter
from samosa.core.proposal import AdaptiveProposal

from samosa.samplers.single_chain import MCMCsampler

from samosa.utils.post_processing import get_position_from_states

# Load samples from the output directory
with open(f'examples/banana/samples.pkl', "rb") as f:
    samples = pickle.load(f)

# Take only the first 5000 samples for the transport map
samples = samples[:5000]

# Now start the Transport map sampling
from samosa.kernels.metropolis_transport import TransportMetropolisHastingsKernel
from samosa.maps.triangular import LowerTriangularMap

# Define the model
model = banana_model

# Define your map using the samples 
map = LowerTriangularMap(samples, 2, 2, 1, 15000, 5000)

# Just add a map componenet to the kernel
kernel = TransportMetropolisHastingsKernel(model, map)

# Redefine the sampler
proposal = GaussianRandomWalk(mu=np.zeros((2,1)), sigma=np.eye(2))

# Define the sampler
sampler = MCMCsampler(model, kernel, proposal, initial_position=samples[-1].position, n_iterations=50000)
ar2 = sampler.run('examples/banana_transport')
print("Acceptance rate:", ar2)

samples_transport = sampler.load_samples('examples/banana_transport')

# Get the positions of the samples
burnin = 0.25
positions = get_position_from_states(samples_transport, burnin)
print("Positions shape:", positions.shape)

print("Mean of the samples:", np.mean(positions, axis=1))
print("Standard deviation of the samples:", np.std(positions, axis=1))

# Plot the scatter plot of the samples
from samosa.utils.post_processing import scatter_matrix, plot_trace, plot_lag
fig, _, _ = scatter_matrix([positions])
plt.savefig('examples/banana_transport/scatter.png')

# Plot the trace of the samples
fig, _ = plot_trace(positions)
plt.savefig('examples/banana_transport/trace.png')

# Plot the lag of the samples
fig, _ = plot_lag(positions)
plt.savefig('examples/banana_transport/lag.png')
