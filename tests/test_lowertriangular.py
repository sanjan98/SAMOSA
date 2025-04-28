import pytest
import numpy as np
from samosa.core.state import ChainState
from samosa.maps.triangular import LowerTriangularMap

@pytest.fixture
def banana_samples():
    """Generate banana-shaped samples for testing."""
    # Sample 100 points from standard Gaussian
    num_points = 100
    np.random.seed(42)  # For reproducibility
    r = np.random.randn(2, num_points)
    
    # Create banana points x from r using the analytical transformation
    x = np.zeros_like(r)
    x[0] = r[0]
    x[1] = r[1] + r[0]**2
    
    # Create ChainState samples from x
    samples = []
    for i in range(num_points):
        position = x[:, i].reshape(-1, 1)
        samples.append(ChainState(
            position=position,
            log_posterior=None
        ))
    
    return samples, x, r

@pytest.fixture
def triangular_map(banana_samples):
    """Create a LowerTriangularMap instance for testing."""
    samples, _, _ = banana_samples
    return LowerTriangularMap(samples, dim=2, total_order=2)

def test_forward_transformation_normalization(triangular_map, banana_samples):
    """Test that forward transformation normalizes data to standard normal."""
    _, x, _ = banana_samples
    
    # Apply the forward transformation to x
    r_map, _ = triangular_map.forward(x)
    
    # Check the mean and standard deviation of the transformed points
    r_mean = np.mean(r_map, axis=1, keepdims=True)
    r_std = np.std(r_map, axis=1, keepdims=True)
    
    # Assert that the mean of the transformed points is close to 0
    assert np.allclose(r_mean, 0, atol=1e-2), f"Transformed mean is not close to 0: {r_mean.flatten()}"
    # Assert that the standard deviation of the transformed points is close to 1
    assert np.allclose(r_std, 1, atol=1e-2), f"Transformed std is not close to 1: {r_std.flatten()}"

def test_inverse_transformation_preserves_statistics(triangular_map, banana_samples):
    """Test that inverse transformation preserves original statistics."""
    _, x, _ = banana_samples
    
    # Original statistics
    x_mean = np.mean(x, axis=1, keepdims=True)
    x_std = np.std(x, axis=1, keepdims=True)
    
    # Apply the forward transformation to x
    r_map, _ = triangular_map.forward(x)
    
    # Apply the inverse transformation to r_map
    x_map, _ = triangular_map.inverse(r_map)
    
    # Check statistics of inverse transformed points
    x_map_mean = np.mean(x_map, axis=1, keepdims=True)
    x_map_std = np.std(x_map, axis=1, keepdims=True)
    
    # Assert that statistics are preserved
    assert np.allclose(x_map_mean, x_mean, atol=1e-2), \
        f"Inverse transformed mean is not close to original mean: {x_map_mean.flatten()}"
    assert np.allclose(x_map_std, x_std, atol=1e-2), \
        f"Inverse transformed std is not close to original std: {x_map_std.flatten()}"

def test_log_determinant_consistency(triangular_map, banana_samples):
    """Test that log determinants from forward and inverse transformations have opposite signs."""
    _, x, _ = banana_samples
    
    # Apply the forward transformation to x
    _, log_det_forward = triangular_map.forward(x)
    
    # Apply the forward transformation to get r_map
    r_map, _ = triangular_map.forward(x)
    
    # Apply the inverse transformation to r_map
    _, log_det_inverse = triangular_map.inverse(r_map)
    
    # Assert that log determinants have opposite signs
    assert np.allclose(log_det_forward, -log_det_inverse, rtol=0.1, atol=0.1), \
        "Log determinants do not have opposite signs"
