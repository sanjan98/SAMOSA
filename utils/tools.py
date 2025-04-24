"""
Script housing some helper functions
"""

# Imports
import numpy as np

# For comments on the first four functions see the Gaussian Random Variable Notebook
def lognormpdf(x, mean, cov):
    """Compute log pdf of a multivariate Normal distribution.
    
    Inputs
    ------
    x : (d, N) array
        Points at which to evaluate the log PDF
    mean : (d, 1) or (d,) array
        Mean of the distribution (column vector or 1D array)
    cov : (d, d) array
        Covariance matrix
    
    Returns
    -------
    logpdf : float or (N,) array
        Log PDF value(s) - scalar if N=1, array otherwise
    """
    # Flatten mean to handle both (d,) and (d, 1) inputs
    mean = mean.flatten()  # Now guaranteed to be (d,)
    
    if x.ndim == 1:
        x = x[:, np.newaxis]
    d, N = x.shape
    
    # Precompute normalization term
    det = np.linalg.det(cov)
    preexp = 1.0 / (2.0 * np.pi)**(d/2) / np.sqrt(det)
    
    # Compute deviation using efficient broadcasting
    diff = x - mean[:, np.newaxis]  # Automatic shape (d, N)
    
    # Solve Σ^{-1}(x - μ) and compute quadratic term
    sol = np.linalg.solve(cov, diff)
    inexp = np.einsum("ij,ij->j", diff, sol)
    
    # Log PDF calculation with squeeze for scalar output
    logpdf = np.log(preexp) - 0.5 * inexp
    return logpdf.squeeze()

def batch_normal_pdf(x, mu, cov, logpdf=True):
    """
    Compute the multivariate normal pdf at each x location.
    Dimensions
    ----------
    d: dimension of the problem
    *: any arbitrary shape (a1, a2, ...)
    Parameters
    ----------
    x: (*, d) location to compute the multivariate normal pdf
    mu: (*, d) mean values to use at each x location 
    cov: (*, d, d) covariance matrix
    Returns
    -------
    pdf: (*) the multivariate normal pdf at each x location
    """
    # Make some checks on input
    x = np.atleast_1d(x)
    mu = np.atleast_1d(mu)
    cov = np.atleast_1d(cov)
    dim = cov.shape[-1]

    # 1-D case
    if len(cov.shape) == 1:
        cov = cov[:, np.newaxis]    # (1, 1)
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    if len(mu.shape) == 1:
        mu = mu[np.newaxis, :]

    assert cov.shape[-1] == cov.shape[-2] == dim
    assert x.shape[-1] == mu.shape[-1] == dim

    # Normalizing constant (scalar)
    preexp = 1 / ((2*np.pi)**(dim/2) * np.linalg.det(cov)**(1/2))

    # Can broadcast x - mu with x: (1, Nr, Nx, d) and mu: (Ns, Nr, Nx, d)
    diff = x - mu

    # In exponential
    diff_col = diff.reshape((*diff.shape, 1))  # (*, d, 1)
    diff_row = diff.reshape((*diff.shape[:-1], 1, diff.shape[-1]))  # (*, 1, d)
    inexp = np.squeeze(diff_row @ np.linalg.inv(cov) @ diff_col, axis=(-1, -2))  # (*, 1, d) x (*, d, 1) = (*, 1, 1)

    # Compute pdf
    pdf = np.log(preexp) + (-1/2)*inexp if logpdf else preexp * np.exp(-1 / 2 * inexp)

    return pdf.astype(np.float32)


def batch_normal_sample(mean, cov, size: "tuple | int" = ()):
    """
    Batch sample multivariate normal distributions.
    https://stackoverflow.com/questions/69399035/is-there-a-way-of-batch-sampling-from-numpys-multivariate-normal-distribution-i
    Arguments:
        mean: expected values of shape (…M, D)
        cov: covariance matrices of shape (…M, D, D)
        size: additional batch shape (…B)
    Returns: samples from the multivariate normal distributions
             shape: (…B, …M, D)
    """
    # Make some checks on input
    mean = np.atleast_1d(mean)
    cov = np.atleast_1d(cov)
    dim = cov.shape[0]

    # 1-D case
    if dim == 1:
        cov = cov[:, np.newaxis]    # (1, 1)
    if len(mean.shape) == 1:
        mean = mean[np.newaxis, :]

    assert cov.shape[0] == cov.shape[1] == dim
    assert mean.shape[-1] == dim

    size = (size, ) if isinstance(size, int) else tuple(size)
    shape = size + np.broadcast_shapes(mean.shape, cov.shape[:-1])
    X = np.random.standard_normal((*shape, 1)).astype(np.float32)
    L = np.linalg.cholesky(cov)
    sample = (L @ X).reshape(shape) + mean
    if dim == 1:
        sample = np.squeeze(sample, axis=-1)
    return sample

def normal_sample(mean, cov, nsamples=1):
    """Generate nsamples from a multivariate Normal distribution

    Inputs
    ------
    mean: (d, ) Mean of distribution
    cov: (d, d) Covariance of distribution
    nsamples: (int) Number of samples
    
    Returns
    -------
    samples: (d, nsamples) Column vector of samples
    """

    # Generate standard normal samples
    mean = np.atleast_1d(mean)
    cov = np.atleast_1d(cov)
    dim = len(mean)
    standard_normal_samples = np.random.randn(dim, nsamples)
    # Apply Cholesky factorization on the covariance matrix
    try:
        cholesky_factor = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = nearest_positive_definite(cov)
        cholesky_factor = np.linalg.cholesky(cov)
    # Generate the samples
    samples = mean[:,np.newaxis] + cholesky_factor @ standard_normal_samples
    samples = np.squeeze(samples)
    return samples

def sample_multivariate_gaussian(mu: np.ndarray, sigma: np.ndarray, N: int = 1) -> np.ndarray:
    """
    Sample from a multivariate Gaussian distribution.

    Inputs:
    ------
        mu: (d, 1) mean vector
        sigma: (d, d) covariance matrix
        N: int, number of samples to generate

    Returns:
    -------
        samples: (d, N) array of samples
    """
    # Ensure mu is a column vector
    mu = np.atleast_2d(mu)
    if mu.shape[1] != 1:
        raise ValueError("mu must be a column vector of shape (d, 1)")
    if mu.shape[0] != sigma.shape[0]:
        raise ValueError("mu and sigma must have the same number of rows (d)")
    if mu.shape[0] != sigma.shape[1]:
        raise ValueError("sigma must be a square matrix of shape (d, d)")
    
    # Convert (d, 1) column vector to 1D array required by NumPy
    mu_flat = mu.flatten()  # Shape becomes (d,)
    
    # Generate samples (returns array of shape (N, d))
    samples = np.random.multivariate_normal(mu_flat, sigma, N)
    
    # Transpose to get (d, N) output format
    return samples.T

def banana_logpdf(x,a=1.0,b=100.0):
    logpdf = (a-x[0])**2 + b * (x[1] - x[0]**2)**2
    #logpdf = np.exp(-0.5 * (x[0])**2 + x[1]**2) - np.exp(-0.5 / 1.0 * (x[0]**2 + x[1]**2))
    #logpdf = np.log(logpdf)
    #logpdf = (np.sin(10*x[0]*x[1]) + x[1]**2)*4
    return -logpdf
    #return logpdf
    
def laplace_approx(x0, logpost, optmethod):
    """Perform the laplace approximation, returning the MAP point and an approximation of the covariance
    :param x0: (nparam, ) array of initial parameters
    :param logpost: f(param) -> log posterior pdf

    :returns map_point: (nparam, ) MAP of the posterior
    :returns cov_approx: (nparam, nparam), covariance matrix for Gaussian fit at MAP
    """
    # Gradient free method to obtain optimum
    neg_post = lambda x: -logpost(x)
    # Use differential evolution to find a robust global minimum
    # result = scipy.optimize.differential_evolution(neg_post, bounds, strategy='best1bin', maxiter=50, popsize=15, tol=1e-6, disp=True)
    res = scipy.optimize.minimize(neg_post, x0)#, method=optmethod, tol=1e-6, options={'maxiter': 1000, 'disp': True})
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
    print('--------------------First optimization done---------------------------------')
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
    # Gradient method which also approximates the inverse of the hessian
    res = scipy.optimize.minimize(neg_post, res.x*0.95, method=optmethod, tol=1e-6, options={'maxiter': 5000, 'disp': True})
    map_point = res.x
    cov_approx = res.hess_inv
    return map_point, cov_approx

def log_banana(x,co):
    if (len(x.shape) == 1):
        x = x[np.newaxis, :]
    N, d = x.shape
    x1p = x[:, 0]
    x2p = x[:, 1] + (np.square(x[:, 0]) + 1)
    xp = np.concatenate((x1p[:, np.newaxis], x2p[:, np.newaxis]), axis=1)
    sigma = np.array([[1, 0.9], [0.9, 1]])
    mu = np.array([0, 0])
    preexp = 1.0 / (2.0 * np.pi)**(d/2) / np.linalg.det(sigma)**0.5
    diff = xp - np.tile(mu[np.newaxis, :], (N, 1))
    sol = np.linalg.solve(sigma, diff.T)
    inexp = np.einsum("ij,ij->j", diff.T, sol)
    co+=1
    return np.log(preexp) - 0.5 * inexp, co

def lognormpdf_univariate(x, mean, cov):
    """Compute the log pdf of a univariate Normal distribution
    
    Inputs
    ------
    x : (float) variable of interest
    mean : (float) mean of the distribution
    cov  : (float) covariance of the distribution
    
    Returns
    -------
    logpdf: (float) log pdf value
    """

    preexp = 1.0 / (2.0 * np.pi)**(0.5) / (cov)**0.5
    diff = x - mean
    inexp = diff**2/cov
    logpdf = np.log(preexp) - 0.5 * inexp
    return logpdf

def invgamma_univariate(x, alpha, beta):
    """Compute the log pdf of a univariate Inverse Gamma distribution
    
    Inputs
    ------
    x : (float) variable of interest
    alpha : (float) shape parameter
    beta  : (float) scale parameter
    
    Returns
    -------
    logpdf: (float) log pdf value
    """

    logpdf = alpha*np.log(beta) - scipy.special.gammaln(alpha) - (alpha+1)*np.log(x) - beta/x
    return logpdf

def nearest_positive_definite(A):
    """Find the nearest positive definite matrix to A."""
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T * s, V)

    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if is_positive_definite(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def is_positive_definite(A):
    """Check if a matrix A is positive definite by attempting Cholesky decomposition."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
    
def batched_variance(data, batch_size):
    """Calculates the variance of a dataset using a batched approach."""

    n_batches = int(np.ceil(len(data) / batch_size))
    batch_variances = []

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(data))
        batch = data[start:end]
        batch_variances.append(np.var(batch, ddof=1))  # Sample variance

    return np.mean(batch_variances)
