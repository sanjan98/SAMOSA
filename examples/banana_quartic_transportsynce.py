# Some imports
import numpy as np
import sys
import os
from samosa.utils.tools import log_banana, log_quartic, sample_multivariate_gaussian
from samosa.utils.post_processing import scatter_matrix
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

def quartic_model(x: np.ndarray) -> Dict[str, Any]:
    """
    Quartic model function
    """
    output = {}
    # Just use the log_banana function to compute the log posterior
    log_posterior = log_banana(x, mu = np.array([[1], [1]]), sigma = np.array([[2, 0.5], [0.5, 1]])) 

    output['log_posterior'] = log_posterior

    # If you want to compute the log likelihood and log prior separately, you can do it like this
    # log_likelihood = <Some custom function>(x)
    # log_prior = <Some custom function>(x)
    # output['log_likelihood'] = log_likelihood
    # output['log_prior'] = log_prior

    # If you want to compute the qoi, cost_model_output etc. you can do it like this
    cost = 4
    qoi = np.sum(x, axis=0)

    output['cost'] = cost
    output['qoi'] = qoi

    return output

# -----------------------------
# Now lets run Transport SYNCE MCMC
# -----------------------------

from samosa.samplers.coupled_chain import coupledMCMCsampler
from samosa.kernels.synce_transport import TransportSYNCEKernel
from samosa.proposals.gaussianproposal import GaussianRandomWalk, IndependentProposal
from samosa.proposals.adapters import HaarioAdapter, GlobalAdapter
from samosa.core.proposal import AdaptiveProposal
from samosa.maps.triangular import LowerTriangularMap
from samosa.utils.post_processing import load_coupled_samples, get_position_from_states


# Define the model
model_coarse = banana_model
model_fine = quartic_model

# proposal = GaussianRandomWalk(mu=np.zeros((2,1)), sigma=np.eye(2))
proposal = IndependentProposal(mu=np.zeros((2,1)), sigma=2.38**2 / 2 * np.eye(2))
# adapter = GlobalAdapter(ar = 0.44, adapt_start=1000, adapt_end=10000)
# adaptive_proposal = AdaptiveProposal(proposal, adapter)

# Define map based on some samples
samples = load_coupled_samples('examples/banana_quartic_synce')
samples_coarse = samples[0][:20000]
samples_fine = samples[1][:20000]

# Define your map using the samples 
map_coarse = LowerTriangularMap(2, 2, 1, 20000, 5000)
map_coarse.adapt(samples_coarse, force_adapt=True)

map_fine = LowerTriangularMap(2, 2, 1, 20000, 5000)
map_fine.adapt(samples_fine, force_adapt=True)

# Check how the maps look like
# Generate 5000 random samples for visualization
r = sample_multivariate_gaussian(np.zeros((2, 1)), np.eye(2), 5000)

# Visualize the random samples using the coarse and fine maps
mapped_samples_coarse, _ = map_coarse.inverse(r)
mapped_samples_fine, _ = map_fine.inverse(r)

fig, _, _ = scatter_matrix([mapped_samples_coarse, mapped_samples_fine])
fig.savefig('examples/banana_quartic_transportsynce/mapped_samples.png')

# Plot the mapped samples
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(mapped_samples_coarse[0, :], mapped_samples_coarse[1, :], alpha=0.5, label="Coarse Map")
ax[0].set_title("Mapped Samples (Coarse)")
ax[0].legend()

ax[1].scatter(mapped_samples_fine[0, :], mapped_samples_fine[1, :], alpha=0.5, label="Fine Map")
ax[1].set_title("Mapped Samples (Fine)")
ax[1].legend()

plt.tight_layout()
plt.savefig('examples/banana_quartic_synce/mapped_samples.png')
plt.close(fig)

kernel = TransportSYNCEKernel(model_coarse, model_fine, map_coarse, map_fine)

# Define the sampler
sampler = coupledMCMCsampler(kernel, proposal, proposal, initial_position_coarse=np.zeros((2, 1)), initial_position_fine=np.zeros((2, 1)), n_iterations=50000, print_iteration=1000, save_iteration=60000) 
ar1, ar2 = sampler.run('examples/banana_quartic_transportsynce')

print("Acceptance rate:", ar1)
print("Acceptance rate:", ar2)

# Load samples from the output directory
samples_coarse, samples_fine = load_coupled_samples('examples/banana_quartic_transportsynce')

# Get the positions of the samples
burnin = 0.2
positions_coarse = get_position_from_states(samples_coarse, burnin)
positions_fine = get_position_from_states(samples_fine, burnin)
print("Positions shape:", positions_coarse.shape)

print("Mean of the coarse samples:", np.mean(positions_coarse, axis=1))
print("Standard deviation of the samples:", np.std(positions_fine, axis=1))

print("Mean of the fine samples:", np.mean(positions_fine, axis=1))
print("Standard deviation of the samples:", np.std(positions_fine, axis=1))

# Compute the correlation between positions_fine and positions_coarse
correlations = []
for i in range(positions_fine.shape[0]):
    correlation_matrix = np.corrcoef(positions_fine[i], positions_coarse[i])
    correlation = correlation_matrix[0, 1]
    correlations.append(correlation)
    print(f"Correlation in dimension {i + 1}: {correlation}")

# Plot the scatter plot of the samples
from samosa.utils.post_processing import scatter_matrix, plot_trace, plot_lag
fig, _, _ = scatter_matrix([positions_coarse, positions_fine])
plt.savefig('examples/banana_quartic_transportsynce/scatter.png')
plt.close(fig)

# Plot the trace of the samples
fig, _ = plot_trace(positions_coarse)
plt.savefig('examples/banana_quartic_transportsynce/trace_coarse.png')
plt.close(fig)
fig, _ = plot_trace(positions_fine)
plt.savefig('examples/banana_quartic_transportsynce/trace_fine.png')
plt.close(fig)

# Plot the lag of the samples
fig, _ = plot_lag(positions_coarse)
plt.savefig('examples/banana_quartic_transportsynce/lag_coarse.png')
plt.close(fig)
fig, _ = plot_lag(positions_fine)
plt.savefig('examples/banana_quartic_transportsynce/lag_fine.png')
plt.close(fig)
