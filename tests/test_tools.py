import numpy as np
import pytest
from scipy import stats
from contextlib import contextmanager

# Import the functions to test
# Note: You'll need to adjust this import path based on your actual module structure
from samosa.utils.tools import (
    lognormpdf, 
    sample_multivariate_gaussian, 
    laplace_approx, 
    log_banana, 
    nearest_positive_definite, 
    is_positive_definite, 
    batched_variance
)

@contextmanager
def not_raises():
    """Context manager to check that no exception is raised."""
    try:
        yield
    except Exception as e:
        raise pytest.fail(f"Unexpected exception raised: {e}")

class TestLognormpdf:
    def test_scalar_output(self):
        """Test lognormpdf with a single point."""
        x = np.array([[1.0], [2.0]])
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        result = lognormpdf(x, mean, cov)
        expected = stats.multivariate_normal.logpdf(x.reshape(-1), mean, cov)
        
        assert isinstance(result, float)
        assert np.isclose(result, expected)
    
    def test_array_output(self):
        """Test lognormpdf with multiple points."""
        x = np.array([[1.0, 2.0], [2.0, 3.0]])  # 2 points in 2D
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        
        result = lognormpdf(x, mean, cov)
        
        # Compute expected using scipy for verification
        expected = np.array([
            stats.multivariate_normal.logpdf(x[:, 0], mean, cov),
            stats.multivariate_normal.logpdf(x[:, 1], mean, cov)
        ])
        
        assert result.shape == (2,)
        assert np.allclose(result, expected)
        
    def test_column_vector_mean(self):
        """Test lognormpdf with mean as column vector."""
        x = np.array([1.0, 2.0])
        mean = np.array([[0.0], [0.0]])  # Column vector
        cov = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        result = lognormpdf(x, mean, cov)
        expected = stats.multivariate_normal.logpdf(x, mean.flatten(), cov)
        
        assert np.isclose(result, expected)

class TestSampleMultivariateGaussian:
    def test_shape_output(self):
        """Test the shape of output from sample_multivariate_gaussian."""
        mu = np.array([[0.0], [0.0]])
        sigma = np.array([[1.0, 0.5], [0.5, 2.0]])
        N = 100
        
        samples = sample_multivariate_gaussian(mu, sigma, N)
        
        assert samples.shape == (2, N)
    
    def test_sample_statistics(self):
        """Test that sample statistics are close to the distribution parameters."""
        mu = np.array([[3.0], [-2.0]])
        sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
        N = 10000
        
        samples = sample_multivariate_gaussian(mu, sigma, N)
        
        # Check sample mean
        sample_mean = np.mean(samples, axis=1)
        assert np.allclose(sample_mean, mu.flatten(), atol=0.1)
        
        # Check sample covariance
        sample_cov = np.cov(samples)
        assert np.allclose(sample_cov, sigma, atol=0.2)
    
    def test_input_validation(self):
        """Test input validation for sample_multivariate_gaussian."""
        # Test with row vector
        mu_row = np.array([0.0, 0.0])
        sigma = np.array([[1.0, 0.5], [0.5, 2.0]])
        
        with pytest.raises(ValueError):
            sample_multivariate_gaussian(mu_row, sigma)
        
        # Test with incompatible dimensions
        mu_wrong = np.array([[0.0], [0.0], [0.0]])
        
        with pytest.raises(ValueError):
            sample_multivariate_gaussian(mu_wrong, sigma)
        
        # Test with non-square sigma
        sigma_wrong = np.array([[1.0, 0.5], [0.5, 2.0], [0.1, 0.2]])
        mu = np.array([[0.0], [0.0]])
        
        with pytest.raises(ValueError):
            sample_multivariate_gaussian(mu, sigma_wrong)

class TestLaplaceApprox:
    def test_simple_gaussian(self):
        """Test Laplace approximation on a simple Gaussian posterior."""
        # Define a simple log posterior (negative quadratic function)
        def logpost(x):
            return -0.5 * np.sum(x**2)
        
        x0 = np.array([[1.0], [1.0]])  # Initial guess])
        
        # For this simple case, MAP should be at origin and covariance should be identity
        map_point, cov_approx = laplace_approx(x0, logpost, 'BFGS')
        
        assert np.allclose(map_point, np.zeros_like(x0), atol=1e-4)
        assert np.allclose(np.diag(cov_approx), np.ones_like(x0), atol=0.1)

class TestLogBanana:
    def test_banana_transform(self):
        """Test that log_banana correctly transforms and evaluates the PDF."""
        # Simple test point
        x = np.array([[1.0], [2.0]])
        mu = np.array([0.0, 0.0])
        sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        # Compute the transformed point manually
        y0 = x[0]
        y1 = x[1] - y0**2
        y = np.array([y0, y1])
        
        # Expected log PDF at the transformed point
        expected = stats.multivariate_normal.logpdf(y.reshape(-1), mu, sigma)
        
        # Actual log PDF from our function
        result = log_banana(x, mu, sigma)
        
        assert np.isclose(result, expected)
    
    def test_multiple_points(self):
        """Test log_banana with multiple points."""
        x = np.array([[1.0, 2.0], [2.0, 5.0]])  # 2 points in 2D
        mu = np.array([0.0, 0.0])
        sigma = np.array([[1.0, 0.5], [0.5, 2.0]])
        
        result = log_banana(x, mu, sigma)
        
        # Manually compute transformed points
        y0 = x[0, :]
        y1 = x[1, :] - y0**2
        
        expected = np.array([
            stats.multivariate_normal.logpdf([y0[0], y1[0]], mu, sigma),
            stats.multivariate_normal.logpdf([y0[1], y1[1]], mu, sigma)
        ])
        
        assert result.shape == (2,)
        assert np.allclose(result, expected)

class TestPositiveDefinite:
    def test_is_positive_definite(self):
        """Test is_positive_definite function."""
        # Test with positive definite matrix
        matrix_pd = np.array([[2.0, 0.5], [0.5, 2.0]])
        assert is_positive_definite(matrix_pd) == True
        
        # Test with non-positive definite matrix
        matrix_non_pd = np.array([[1.0, 2.0], [2.0, 1.0]])
        assert is_positive_definite(matrix_non_pd) == False
    
    def test_nearest_positive_definite(self):
        """Test nearest_positive_definite function."""
        # Test with a matrix that's not positive definite
        matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
        result = nearest_positive_definite(matrix)
        
        # The result should be positive definite
        assert is_positive_definite(result) == True
        
        # The result should be close to the original matrix
        assert np.linalg.norm(result - matrix) < 2.0

class TestBatchedVariance:
    def test_1d_variance(self):
        """Test batched variance for 1D data."""
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])  # 1 dimension, 5 samples
        batch_size = 2
        
        result = batched_variance(data, batch_size)
        expected = np.var(data, axis=1, ddof=1)
        
        assert result.shape == (1,)
        assert np.isclose(result[0], expected[0])
    
    def test_multidimensional_variance(self):
        """Test batched variance for multi-dimensional data."""
        # Create a 3D dataset with 1000 samples
        np.random.seed(42)
        data = np.random.randn(3, 1000)
        batch_size = 100
        
        result = batched_variance(data, batch_size)
        expected = np.var(data, axis=1, ddof=1)
        
        assert result.shape == (3,)
        assert np.allclose(result, expected)
    
    def test_different_batch_sizes(self):
        """Test that different batch sizes give consistent results."""
        np.random.seed(42)
        data = np.random.randn(2, 500)
        
        result1 = batched_variance(data, 50)
        result2 = batched_variance(data, 100)
        result3 = batched_variance(data, 250)
        expected = np.var(data, axis=1, ddof=1)
        
        assert np.allclose(result1, expected)
        assert np.allclose(result2, expected)
        assert np.allclose(result3, expected)
    
    def test_edge_cases(self):
        """Test edge cases for batched variance."""
        # Test with batch size larger than dataset
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = batched_variance(data, 10)
        expected = np.var(data, axis=1, ddof=1)
        assert np.allclose(result, expected)
        
        # Test with batch size equal to dataset size
        result = batched_variance(data, 3)
        assert np.allclose(result, expected)
        
        # Test with batch size of 1
        result = batched_variance(data, 1)
        assert np.allclose(result, expected)