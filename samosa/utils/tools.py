"""
Script housing some helper functions
"""

# Imports
import numpy as np
import scipy
from typing import Callable, Optional

def lognormpdf(x: np.ndarray, mean: Optional[np.ndarray] = None, cov: Optional[np.ndarray] = None) -> np.ndarray:

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

    # Convert scalars to arrays for unified handling
    if np.isscalar(x):
        x = np.array([[x]], dtype=float)
    elif isinstance(x, np.ndarray) and x.ndim == 1:
        x = x[:, np.newaxis]  # shape (d, N)
    
    d, N = x.shape

    if mean is None:
        mean = np.zeros((d, 1), dtype=float)
    if cov is None:
        cov = np.eye(d, dtype=float)
        
    if np.isscalar(mean):
        mean = np.array([mean], dtype=float)
    else:
        mean = np.asarray(mean).flatten()
        
    if np.isscalar(cov):
        cov = np.array([[cov]], dtype=float)
    else:
        cov = np.asarray(cov)
    
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

    # If N=1, return a scalar
    if N == 1:
        return logpdf.item()
    # Otherwise, return as a 1D array
    else:
        return logpdf.flatten()

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
    
def laplace_approx(x0: np.ndarray, logpost: Callable, optmethod: str):
    """Perform the laplace approximation, returning the MAP point and an approximation of the covariance matrix.

    Parameters
    ----------
    x0 : (d, 1) array
        Initial guess for the MAP point
    logpost : callable
        Function to compute the log posterior
    optmethod : str
        Optimization method to use (e.g., 'L-BFGS-B', 'BFGS', etc.)

    Returns
    -------
    x_map : (d, 1) array
        MAP point
    cov_approx : (d, d) array
        Approximation of the covariance matrix
    """

    # Enforce x0 to be a column vector
    assert x0.ndim == 2 and x0.shape[1] == 1, "x0 must be a column vector"
    d = x0.shape[0]

    # Change x0 to be a 1D array for optimization
    x0 = x0.flatten()
    
    # Gradient free method to obtain optimum
    neg_post = lambda x: -logpost(x)
    
    # Initial optimization using a gradient-free method
    res = scipy.optimize.minimize(neg_post, x0)
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
    print('--------------------First optimization done---------------------------------')
    print('----------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------')
    # Gradient method which also approximates the inverse of the hessian
    res = scipy.optimize.minimize(neg_post, res.x*0.95, method=optmethod, tol=1e-6, options={'maxiter': 5000, 'disp': True})
    map_point = res.x
    # Make map point a column vector
    map_point = map_point[:, np.newaxis] if map_point.ndim == 1 else map_point
    cov_approx = res.hess_inv
    return map_point, cov_approx

def log_banana(x: np.ndarray, mu: Optional[np.ndarray] = None, sigma: Optional[np.ndarray] = None, shift: Optional[float] = 0.0) -> np.ndarray:
    """
    Log pdf of the banana distribution.

    Parameters
    ----------
    x : (d, N) array
        Points at which to evaluate the log PDF
    mu : (d, 1) or (d,) array
        Mean of the distribution (column vector or 1D array)
    sigma : (d, d) array
        Covariance matrix
    
    Returns
    -------
    logpdf : float or (N,) array
        Log PDF value(s) - scalar if N=1, array otherwise
    """

    # Assert that x is 2D
    assert x.ndim == 2, "x must be a 2D array"

    # Check if mu is None and set to zero if so
    if mu is None:
        mu = np.zeros((x.shape[0], 1))
    
    # Check if sigma is None and set to identity if so
    if sigma is None:
        sigma = np.eye(x.shape[0])

    # Flatten mean to handle both (d,) and (d, 1) inputs
    mu = mu.flatten()  # Now guaranteed to be (d,)

    # Check if x is 1D and reshape if necessary
    if x.ndim == 1:
        x = x[:, np.newaxis]
    d, N = x.shape

    # Compute the transformation
    x0 = x[0, :] - shift; x1 = x[1, :]
    y0 = x0; y1 = x1 + y0**2

    y = np.vstack((y0, y1))

    # Compute the log PDF using the multivariate normal distribution
    logpdf = lognormpdf(y, mu, sigma)

    return logpdf

def log_quartic(x: np.ndarray, mu: Optional[np.ndarray] = None, sigma: Optional[np.ndarray] = None, shift: Optional[float] = 0.0) -> np.ndarray:
    """
    Log pdf of the quartic-shaped distribution.

    Parameters
    ----------
    x : (d, N) array
        Points at which to evaluate the log PDF
    mu : (d, 1) or (d,) array
        Mean of the distribution (column vector or 1D array)
    sigma : (d, d) array
        Covariance matrix
    
    Returns
    -------
    logpdf : float or (N,) array
        Log PDF value(s) - scalar if N=1, array otherwise
    """

    assert x.ndim == 2, "x must be a 2D array"

    # Handle default parameters
    if mu is None:
        mu = np.zeros((x.shape[0], 1))
    if sigma is None:
        sigma = np.eye(x.shape[0])

    mu = mu.flatten()  # Ensure mu is (d,)

    # Reshape x if 1D
    if x.ndim == 1:
        x = x[:, np.newaxis]
    d, N = x.shape

    # Apply quartic transformation
    x0 = x[0, :] - shift; x1 = x[1, :]
    y0 = x0; y1 = x1 + x0**2 + x0**4  # Quartic term defines curvature
    y = np.vstack((y0, y1))

    # Compute log-PDF
    logpdf = lognormpdf(y, mu, sigma)
    
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

def is_positive_definite(A: np.ndarray) -> bool:
    """
    Check if a matrix A is positive definite by attempting Cholesky decomposition.
    
    Parameters
    ----------
    A : (d, d) array
        Matrix to check for positive definiteness
    
    Returns
    -------
    is_pd : bool
        True if A is positive definite, False otherwise
    """
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
    
def batched_variance(data: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
    """
    Calculates the variance of a dataset using a batched approach.
    
    Parameters
    ----------
    data : (d, N) array
        Input data for which to compute the variance, where d is the number of dimensions
        and N is the number of samples
    batch_size : int
        Size of each batch to process
    
    Returns
    -------
    variance : np.ndarray
        Array of shape (d,) containing the variance for each dimension
    """
    # For (d, N) data, we need to compute variance along axis=1
    d, n_samples = data.shape
    
    if batch_size is None:
        batch_size = n_samples // 10

    n_batches = int(np.ceil(n_samples / batch_size))
    
    # Initialize arrays for the running calculation
    count = np.zeros(d)
    mean = np.zeros(d)
    M2 = np.zeros(d)  # Sum of squared differences from the mean
    
    # Process each batch
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_samples)
        batch = data[:, start:end]
        batch_size_actual = end - start
        
        # Update running statistics using Welford's online algorithm for each dimension
        batch_mean = np.mean(batch, axis=1)  # Mean for each dimension
        batch_var = np.var(batch, axis=1, ddof=0) * batch_size_actual  # Unnormalized variance
        
        # Combine with previous batches (for each dimension)
        new_count = count + batch_size_actual
        delta = batch_mean - mean
        mean = mean + delta * (batch_size_actual / new_count)
        M2 = M2 + batch_var + delta**2 * count * batch_size_actual / new_count
        count = new_count
    
    # Calculate the variance with Bessel's correction
    variance = np.divide(M2, count - 1, out=np.zeros_like(M2), where=count > 1)
    
    return variance