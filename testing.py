import numpy as np
import mcmc.utils.mcmc_utils_and_plot as utils

# Define the mean vector and covariance matrix
mean = np.array([[0], [0]])
covariance = np.array([[1, 0], [0, 1]])

print(mean.shape)
print(covariance.shape)

# Generate samples from the multivariate Gaussian distribution
num_samples = 1000
samples = utils.sample_multivariate_gaussian(mean, covariance)
print(samples.shape)
pdf = utils.lognormpdf(samples, mean, covariance)
# Print the shape of the samples array
print(pdf)