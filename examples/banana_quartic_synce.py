# Some imports
import numpy as np
from samosa.utils.tools import log_banana, log_quartic
from samosa.utils.post_processing import scatter_matrix
from typing import Any, Dict
import matplotlib.pyplot as plt

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
    log_posterior = log_banana(x, mu = np.array([[1.0], [1.0]]), sigma = np.array([[2, 0.5],[0.5, 1]])) 

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

# Plot some densities first
# Lets plot to see how our banana model looks like
# Define the grid
x1 = np.linspace(-5, 5, 1000)
x2 = np.linspace(-5, 5, 1000)
X1, X2 = np.meshgrid(x1, x2)

# Define the inputs
x = np.vstack([X1.ravel(), X2.ravel()])

# Compute the log posterior
output_coarse = banana_model(x)
output_fine = quartic_model(x)

# Reshape the output
log_posterior_coarse = output_coarse['log_posterior'].reshape(X1.shape)
posterior_coarse = np.exp(log_posterior_coarse)  # Convert log posterior to posterior
log_posterior_fine = output_fine['log_posterior'].reshape(X1.shape)
posterior_fine = np.exp(log_posterior_fine)  # Convert log posterior to posterior

# Create subplots for Source, Target, and Reference
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

label_fontsize = 26
title_fontsize = 15
tick_fontsize = 24
legend_fontsize = 24

# Plot Source PDF
c1 = axes[0].contour(X1, X2, posterior_coarse, levels=5, cmap='Reds')
axes[0].set_title('Coarse Density', fontsize=title_fontsize)
axes[0].set_xlabel('X', fontsize=label_fontsize)
axes[0].set_ylabel('Y', fontsize=label_fontsize)

# Plot Target PDF
c3 = axes[1].contour(X1, X2, posterior_fine, levels=5, cmap='Greens')
axes[1].set_title('Fine Density', fontsize=title_fontsize)
axes[1].set_xlabel('X', fontsize=label_fontsize)

# Set tick parameters for all subplots
for ax in axes:
    ax.tick_params(axis='both', labelsize=tick_fontsize)

# Adjust layout
plt.tight_layout()

# Save the figure using the specified image format
plt.savefig('examples/banana_quartic_synce/density.png')
plt.close()

# -----------------------------
# Now lets run SYNCE MCMC
# -----------------------------

from samosa.samplers.coupled_chain import coupledMCMCsampler
from samosa.kernels.synce import SYNCEKernel
from samosa.proposals.gaussianproposal import GaussianRandomWalk
from samosa.proposals.adapters import HaarioAdapter, GlobalAdapter
from samosa.core.proposal import AdaptiveProposal
from samosa.utils.post_processing import load_coupled_samples, get_position_from_states


# Define the model
model_coarse = banana_model
model_fine = quartic_model

proposal = GaussianRandomWalk(mu=np.zeros((2,1)), sigma=np.eye(2))
adapter = GlobalAdapter(ar = 0.44, adapt_start=1000, adapt_end=10000)
# adapter = HaarioAdapter(scale=2.38**2/2, adapt_start=1000, adapt_end=10000)
adaptive_proposal = AdaptiveProposal(proposal, adapter)

kernel = SYNCEKernel(model_coarse, model_fine, w=0.2, resync_type='independent')

# Load samples from the output directory for restart
# restart_coarse, restart_fine = load_coupled_samples('examples/banana_quartic_synce', iteration=20000)

# Define the sampler
sampler = coupledMCMCsampler(model_coarse, model_fine, kernel, adaptive_proposal, adaptive_proposal, initial_position_coarse=np.zeros((2, 1)), initial_position_fine=np.zeros((2, 1)), n_iterations=50000, print_iteration=1000, save_iteration=10000)#, restart_coarse=restart_coarse, restart_fine=restart_fine) 
ar1, ar2 = sampler.run('examples/banana_quartic_synce')

print("Acceptance rate:", ar1)
print("Acceptance rate:", ar2)

# Load samples from the output directory
samples_coarse, samples_fine = load_coupled_samples('examples/banana_quartic_synce')

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
plt.savefig('examples/banana_quartic_synce/scatter.png')
plt.close(fig)

# Plot the trace of the samples
fig, _ = plot_trace(positions_coarse)
plt.savefig('examples/banana_quartic_synce/trace_coarse.png')
plt.close(fig)
fig, _ = plot_trace(positions_fine)
plt.savefig('examples/banana_quartic_synce/trace_fine.png')
plt.close(fig)

# Plot the lag of the samples
fig, _ = plot_lag(positions_coarse)
plt.savefig('examples/banana_quartic_synce/lag_coarse.png')
plt.close(fig)
fig, _ = plot_lag(positions_fine)
plt.savefig('examples/banana_quartic_synce/lag_fine.png')
plt.close(fig)
