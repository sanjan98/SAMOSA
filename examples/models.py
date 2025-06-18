"""
This script defines two models:
A 'coarse' banana model and a 'fine' quartic banana model.
Analytical transport maps exist for these posteriors

Analytical transport maps
Reference (r1, r2) ~ N(0, I)
Coarse banana: (b1, b2) = (r1 + s1, r2 - (r1 + s1)^2)
Fine quartic banana: (q1, q2) = (r1 + s2, r2 - (r1 + s2)^2 - (r1 + s2)^4)

Between the two posteriors,
(q1, q2) = (b1 - s1 + s2, b2 + b1^2 - (b1 - s1 + s2)^2 - (b1 - s1 + s2)^4)

"""

# Imports
import numpy as np
from samosa.utils.tools import log_banana, log_quartic, lognormpdf
import matplotlib.pyplot as plt

def banana(x: np.ndarray, shift: float = 0.0) -> np.ndarray:
    """
    Log-PDF of a banana distribution.
    """

    assert x.ndim == 2, "Input must be a 2D array."
    log_posterior = log_banana(x, shift=shift)

    output = {}
    output['log_posterior'] = log_posterior
    cost = 1
    qoi = np.sum(x, axis=0)

    output['cost'] = cost
    output['qoi'] = qoi

    return output

def quartic(x: np.ndarray, shift: float = 0.0) -> np.ndarray:
    """
    Log-PDF of a quartic banana distribution.
    """
    
    assert x.ndim == 2, "Input must be a 2D array."
    log_posterior = log_quartic(x, shift=shift)

    output = {}
    output['log_posterior'] = log_posterior
    cost = 2
    qoi = np.sum(x, axis=0)

    output['cost'] = cost
    output['qoi'] = qoi

    return output

if __name__ == "__main__":
    # Grid for plotting
    x1 = np.linspace(-8, 8, 200)
    x2 = np.linspace(-10, 4, 200)
    X1, X2 = np.meshgrid(x1, x2)
    points = np.vstack([X1.ravel(), X2.ravel()])

    # Evaluate log-pdfs
    banana_logpdf = banana(points)['log_posterior'].reshape(X1.shape)
    quartic_logpdf = quartic(points)['log_posterior'].reshape(X1.shape) 
    reference_logpdf = lognormpdf(points, np.zeros((2, 1)), np.eye(2)).reshape(X1.shape)

    plt.figure(figsize=(10, 7))
    cs1 = plt.contour(X1, X2, np.exp(banana_logpdf), levels=5, cmap='Greens', linewidths=2)
    cs2 = plt.contour(X1, X2, np.exp(quartic_logpdf), levels=5, cmap='Reds', linewidths=2)
    cs3 = plt.contour(X1, X2, np.exp(reference_logpdf), levels=5, cmap='Blues', linewidths=2)
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.grid(True)
    plt.savefig('examples/models.png', dpi=300, bbox_inches='tight')
