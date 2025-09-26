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
from samosa.proposals.gaussianproposal import GaussianRandomWalk
from samosa.samplers.single_chain import MCMCsampler
from samosa.utils.post_processing import load_samples
from samosa.utils.post_processing import get_position_from_states, get_reference_position_from_states

# Now start the Transport map sampling
from samosa.kernels.metropolis_transport import TransportMetropolisHastingsKernel
from samosa.maps.triangular import LowerTriangularMap

# Load samples from the output directory
with open(f'examples/banana-delayedrejection/samples.pkl', "rb") as f:
    samples = pickle.load(f)

# Take only the first 5000 samples for the transport map
samples = samples[:10000]

# Define the model
model = banana_model

# Define your map using the samples 
map = LowerTriangularMap(2, 2, 1, 25000, 5000)
map.adapt(samples, force_adapt=True)

# Just add a map componenet to the kernel
kernel = TransportMetropolisHastingsKernel(model, map)

# Redefine the sampler
proposal = GaussianRandomWalk(mu=np.zeros((2,1)), sigma=0.5*np.eye(2))

# Define the output directory
output = 'examples/banana-transport-mpart'

# Define the sampler
sampler = MCMCsampler(kernel, proposal, initial_position=samples[-1].position, n_iterations=50000, save_iteraton=100000, restart=samples)
ar2 = sampler.run(output)
print("Acceptance rate:", ar2)

samples_transport = load_samples(output)

# Get the positions of the samples
burnin = 0.25
positions = get_position_from_states(samples_transport, burnin)
print("Positions shape:", positions.shape)

reference_positions = get_reference_position_from_states(samples_transport, burnin)
print("Reference positions shape:", reference_positions.shape)

print("Mean of the samples:", np.mean(positions, axis=1))
print("Standard deviation of the samples:", np.std(positions, axis=1))

# Plot the scatter plot of the samples
from samosa.utils.post_processing import scatter_matrix, plot_trace, plot_lag
fig, _, _ = scatter_matrix([positions])
plt.savefig(f'{output}/scatter.png')

fig, _, _ = scatter_matrix([reference_positions])
plt.savefig(f'{output}/reference_scatter.png')

# Plot the trace of the samples
fig, _ = plot_trace(positions)
plt.savefig(f'{output}/trace.png')

# Plot the lag of the samples
fig, _ = plot_lag(positions)
plt.savefig(f'{output}/lag.png')

# Plot the log posterior using seaborn
plt.figure(figsize=(10, 8))
import seaborn as sns
sns.set_style("white")

x1 = np.linspace(-4, 4, 100)
x2 = np.linspace(-6, 4, 100)
X1, X2 = np.meshgrid(x1, x2)
x = np.vstack([X1.ravel(), X2.ravel()])

# Compute the log posterior
banana_output = banana_model(x)

# Reshape the output
log_posterior = banana_output['log_posterior'].reshape(X1.shape)
log_posterior = np.exp(log_posterior)  # Convert log posterior to posterior

# Define the inputs
# Create contour plot with fewer levels and no fill
sns.kdeplot(x=None, y=None, fill=False, levels=7, cmap="viridis", linewidths=1.5)

# Since kdeplot expects data to fit a distribution, we need to use plt.contour directly
plt.contour(X1, X2, log_posterior, levels=7, cmap="viridis", linewidths=1.5)

# Add pullback pdf
pullback_pdf = sampler.kernel.map.pullback(x)
plt.contour(X1, X2, pullback_pdf.reshape(X1.shape), levels=7, cmap="plasma", linestyles='dashed', linewidths=1.5)
plt.xlabel('x1', fontsize=12)
plt.ylabel('x2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{output}/pullback_pdf.png')
