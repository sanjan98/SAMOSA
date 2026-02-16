"""
Shared numerical utilities for SAMOSA.

Provides log-PDFs, sampling, Laplace approximation, and matrix helpers.
Sample arrays use (d, N) convention: d dimensions, N samples.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Union

import numpy as np
import scipy.optimize

from samosa.core.model import ModelProtocol

logger = logging.getLogger(__name__)

# Type alias for log-PDF return (scalar when N=1, else 1D array)
LogPdfResult = Union[float, np.ndarray]


def lognormpdf(
    x: np.ndarray,
    mean: Optional[np.ndarray] = None,
    cov: Optional[np.ndarray] = None,
) -> LogPdfResult:
    """
    Compute log PDF of a multivariate normal distribution.

    Uses log-space and Cholesky/solve for numerical stability.
    Supports scalar or batch evaluation.

    Parameters
    ----------
    x : numpy.ndarray of shape (d, N)
        Points at which to evaluate the log PDF.
    mean : numpy.ndarray of shape (d, 1) or (d,), optional
        Mean of the distribution. Defaults to zero.
    cov : numpy.ndarray of shape (d, d), optional
        Covariance matrix. Defaults to identity.

    Returns
    -------
    logpdf : float or numpy.ndarray of shape (N,)
        Log PDF value(s). Scalar if N=1, otherwise 1D array.

    Notes
    -----
    Uses ``np.linalg.slogdet`` for the normalization term and
    ``np.linalg.solve(cov, diff)`` instead of inverting cov.
    """
    if np.isscalar(x):
        x = np.array([[float(x)]])
    elif isinstance(x, np.ndarray) and x.ndim == 1:
        x = x[:, np.newaxis]

    d, N = x.shape

    if mean is None:
        mean = np.zeros((d, 1), dtype=float)
    if cov is None:
        cov = np.eye(d, dtype=float)

    mean = np.asarray(mean).flatten()
    cov = np.asarray(cov)

    # Log normalization using slogdet for numerical stability
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite.")
    lognorm = -0.5 * (d * np.log(2.0 * np.pi) + logdet)

    diff = x - mean[:, np.newaxis]
    sol = np.linalg.solve(cov, diff)
    inexp = np.einsum("ij,ij->j", diff, sol)
    logpdf = lognorm - 0.5 * inexp

    if N == 1:
        return float(logpdf.item())
    return logpdf.flatten()


def sample_multivariate_gaussian(
    mu: np.ndarray,
    sigma: np.ndarray,
    N: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample from a multivariate Gaussian distribution.

    Parameters
    ----------
    mu : numpy.ndarray of shape (d, 1)
        Mean vector.
    sigma : numpy.ndarray of shape (d, d)
        Covariance matrix (positive definite).
    N : int, optional
        Number of samples. Default is 1.
    rng : numpy.random.Generator, optional
        Random number generator. If None, uses default global generator.

    Returns
    -------
    samples : numpy.ndarray of shape (d, N)
        Samples in (d, N) format.

    Raises
    ------
    ValueError
        If mu is not a column vector or dimensions are incompatible.
    """
    mu = np.atleast_2d(mu)
    if mu.shape[1] != 1:
        raise ValueError("mu must be a column vector of shape (d, 1)")
    if mu.shape[0] != sigma.shape[0] or mu.shape[0] != sigma.shape[1]:
        raise ValueError("sigma must be square (d, d) and match mu dimension d")

    mu_flat = mu.flatten()
    if rng is None:
        samples = np.random.multivariate_normal(mu_flat, sigma, N)
    else:
        samples = rng.multivariate_normal(mu_flat, sigma, N)
    return samples.T


def _get_log_posterior(
    model: Union[ModelProtocol, Callable[..., object]], x: np.ndarray
) -> float:
    """Return log_posterior from model(x); support dict or scalar return."""
    out = model(x)
    if isinstance(out, dict):
        return float(out["log_posterior"])
    return float(out)


def laplace_approx(
    x0: np.ndarray,
    model: Union[ModelProtocol, Callable[..., object]],
    optmethod: str = "BFGS",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Laplace approximation: MAP point and approximate covariance.

    Two-stage optimization: gradient-free then gradient-based refinement.
    Uses the inverse Hessian at the MAP as the covariance approximation.

    Parameters
    ----------
    x0 : numpy.ndarray of shape (d, 1)
        Initial guess for the MAP point.
    model : ModelProtocol or callable
        Model returning log_posterior (dict with 'log_posterior' or scalar).
    optmethod : str, optional
        Optimization method for refinement (e.g. 'BFGS', 'L-BFGS-B').
        Default is 'BFGS'.

    Returns
    -------
    map_point : numpy.ndarray of shape (d, 1)
        MAP point.
    cov_approx : numpy.ndarray of shape (d, d)
        Approximate covariance (inverse Hessian at MAP).

    Notes
    -----
    Prints are replaced with logging. For diagnostics, consider returning
    the optimization result as a third element in a future version.
    """
    if x0.ndim != 2 or x0.shape[1] != 1:
        raise ValueError("x0 must be a column vector of shape (d, 1)")
    d = x0.shape[0]
    x0_flat = x0.flatten()

    def neg_log_post(x: np.ndarray) -> float:
        return -_get_log_posterior(model, x)

    # Gradient-free initial optimization
    res = scipy.optimize.minimize(neg_log_post, x0_flat)
    logger.debug("Laplace approx: first (gradient-free) optimization done.")

    # Gradient-based refinement with Hessian approximation
    res = scipy.optimize.minimize(
        neg_log_post,
        res.x * 0.95,
        method=optmethod,
        tol=1e-6,
        options={"maxiter": 5000, "disp": False},
    )
    map_point = np.asarray(res.x).reshape(-1, 1)
    cov_approx = (
        res.hess_inv
        if hasattr(res, "hess_inv") and res.hess_inv is not None
        else np.eye(d)
    )
    if np.ndim(cov_approx) == 0:
        cov_approx = np.eye(d) * float(cov_approx)
    return map_point, np.atleast_2d(cov_approx)


def log_banana(
    x: np.ndarray,
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    shift: float = 0.0,
) -> LogPdfResult:
    """
    Log PDF of the banana (Rosenbrock-type) distribution.

    Transformation: y0 = x0 - shift, y1 = x1 + y0^2; then log PDF is
    multivariate normal in y with mean mu and covariance sigma.

    Parameters
    ----------
    x : numpy.ndarray of shape (d, N)
        Points at which to evaluate the log PDF.
    mu : numpy.ndarray, optional
        Mean in transformed space. Defaults to zero.
    sigma : numpy.ndarray, optional
        Covariance in transformed space. Defaults to identity.
    shift : float, optional
        Shift for the first coordinate. Default is 0.0.

    Returns
    -------
    logpdf : float or numpy.ndarray of shape (N,)
        Log PDF value(s).
    """
    if x.ndim != 2:
        raise ValueError("x must be a 2D array of shape (d, N)")
    if mu is None:
        mu = np.zeros((x.shape[0], 1))
    if sigma is None:
        sigma = np.eye(x.shape[0])
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma)
    d, N = x.shape
    x0 = x[0, :] - shift
    x1 = x[1, :]
    y0 = x0
    y1 = x1 + y0**2
    y = np.vstack((y0, y1))
    return lognormpdf(y, mu, sigma)


def log_quartic(
    x: np.ndarray,
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    shift: float = 0.0,
) -> LogPdfResult:
    """
    Log PDF of the quartic-shaped distribution.

    Transformation: y0 = x0 - shift, y1 = x1 + x0^2 + x0^4; then
    log PDF is multivariate normal in y.

    Parameters
    ----------
    x : numpy.ndarray of shape (d, N)
        Points at which to evaluate the log PDF.
    mu : numpy.ndarray, optional
        Mean in transformed space. Defaults to zero.
    sigma : numpy.ndarray, optional
        Covariance in transformed space. Defaults to identity.
    shift : float, optional
        Shift for the first coordinate. Default is 0.0.

    Returns
    -------
    logpdf : float or numpy.ndarray of shape (N,)
        Log PDF value(s).
    """
    if x.ndim != 2:
        raise ValueError("x must be a 2D array of shape (d, N)")
    if mu is None:
        mu = np.zeros((x.shape[0], 1))
    if sigma is None:
        sigma = np.eye(x.shape[0])
    mu = np.asarray(mu).flatten()
    d, N = x.shape
    x0 = x[0, :] - shift
    x1 = x[1, :]
    y0 = x0
    y1 = x1 + x0**2 + x0**4
    y = np.vstack((y0, y1))
    return lognormpdf(y, mu, sigma)


def nearest_positive_definite(
    A: np.ndarray,
    tol: Optional[float] = None,
) -> np.ndarray:
    """
    Find the nearest positive definite matrix to A.

    Uses Higham's algorithm (symmetric projection and eigenvalue correction).

    Parameters
    ----------
    A : numpy.ndarray of shape (d, d)
        Input matrix.
    tol : float, optional
        Eigenvalue threshold for positive definiteness. Defaults to
        spacing(norm(A)).

    Returns
    -------
    numpy.ndarray of shape (d, d)
        Nearest positive definite matrix to A.
    """
    A = np.asarray(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T * s, V)
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if is_positive_definite(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A)) if tol is None else tol
    identity_matrix = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 = A3 + identity_matrix * (-mineig * k**2 + spacing)
        k += 1
    return A3


def is_positive_definite(A: np.ndarray) -> bool:
    """
    Check if a matrix is positive definite via Cholesky decomposition.

    Parameters
    ----------
    A : numpy.ndarray of shape (d, d)
        Matrix to check.

    Returns
    -------
    bool
        True if A is positive definite, False otherwise.
    """
    try:
        np.linalg.cholesky(np.asarray(A))
        return True
    except np.linalg.LinAlgError:
        return False


def batched_variance(
    data: np.ndarray,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """
    Variance of data using batched Welford's algorithm.

    Parameters
    ----------
    data : numpy.ndarray of shape (d, N)
        Input data; d dimensions, N samples.
    batch_size : int, optional
        Batch size. Defaults to N // 10.

    Returns
    -------
    variance : numpy.ndarray of shape (d,)
        Sample variance per dimension (ddof=1).
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("data must be 2D with shape (d, N)")
    d, n_samples = data.shape
    if batch_size is None:
        batch_size = max(1, n_samples // 10)

    n_batches = int(np.ceil(n_samples / batch_size))
    count = np.zeros(d)
    mean = np.zeros(d)
    M2 = np.zeros(d)

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_samples)
        batch = data[:, start:end]
        batch_size_actual = end - start
        batch_mean = np.mean(batch, axis=1)
        batch_var = np.var(batch, axis=1, ddof=0) * batch_size_actual
        new_count = count + batch_size_actual
        delta = batch_mean - mean
        mean = mean + delta * (batch_size_actual / new_count)
        M2 = M2 + batch_var + delta**2 * count * batch_size_actual / new_count
        count = new_count

    variance = np.divide(M2, count - 1, out=np.zeros_like(M2), where=count > 1)
    return variance
