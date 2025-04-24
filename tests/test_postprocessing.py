import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
import os

# Import the functions to be tested
from samosa.utils.post_processing import (
    scatter_matrix,
    plot_trace,
    plot_lag,
    autocorrelation,
    effective_sample_size
)

@pytest.fixture
def sample_data():
    """Create synthetic test data in (dim, N) format"""
    np.random.seed(42)
    dim = 3
    N = 1000
    samples = np.random.randn(dim, N)
    return samples

@pytest.fixture
def multiple_samples():
    """Create multiple sample chains in (dim, N) format"""
    np.random.seed(42)
    dim = 3
    N = 1000
    samples1 = np.random.randn(dim, N)
    samples2 = np.random.randn(dim, N) + 1  # Shifted mean
    return [samples1, samples2]

@pytest.fixture
def img_kwargs():
    """Default image parameters"""
    return {
        'label_fontsize': 24,
        'title_fontsize': 20,
        'tick_fontsize': 20,
        'legend_fontsize': 16,
        'img_format': 'png'
    }

def test_scatter_matrix(multiple_samples):
    """Test scatter_matrix function with (dim, N) format data"""
    labels = ["Parameter 1", "Parameter 2", "Parameter 3"]
    fig, axs, gs = scatter_matrix(multiple_samples, labels=labels)
    
    # Check return types
    assert isinstance(fig, Figure)
    assert isinstance(axs, list)
    assert isinstance(gs, GridSpec)
    
    # Verify customization works
    fig.suptitle("Custom Title")
    test_filename = "test_scatter_matrix.png"
    fig.savefig(test_filename, bbox_inches='tight')
    
    # Verify file was created
    assert os.path.exists(test_filename)
    
    # Clean up
    plt.close(fig)
    os.remove(test_filename)

def test_plot_trace(sample_data, img_kwargs):
    """Test plot_trace function with (dim, N) format data"""
    labels = ["Parameter 1", "Parameter 2", "Parameter 3"]
    fig, axs = plot_trace(sample_data, img_kwargs=img_kwargs, labels=labels)
    
    # Check return types
    assert isinstance(fig, Figure)
    assert isinstance(axs, np.ndarray) or isinstance(axs, list)
    
    # Verify dimensions match
    if isinstance(axs, np.ndarray):
        assert len(axs) == sample_data.shape[0]
    else:
        assert len(axs) == sample_data.shape[0]
    
    # Test customization
    if sample_data.shape[0] > 1:
        axs[0].set_title("Custom Title")
    else:
        axs.set_title("Custom Title")
        
    test_filename = "test_trace_plot.png"
    fig.savefig(test_filename)
    
    # Verify file was created
    assert os.path.exists(test_filename)
    
    # Clean up
    plt.close(fig)
    os.remove(test_filename)

def test_autocorrelation(sample_data):
    """Test autocorrelation function with (dim, N) format data"""
    maxlag = 50
    step = 5
    lags, autos = autocorrelation(sample_data, maxlag=maxlag, step=step)
    
    # Check return types
    assert isinstance(lags, np.ndarray)
    assert isinstance(autos, np.ndarray)
    
    # Verify shapes
    expected_lags_len = len(range(0, maxlag, step))
    assert lags.shape == (expected_lags_len,)
    
    # The shape should be (ndim, len(lags)) based on your implementation
    dim = sample_data.shape[0]
    assert autos.shape == (dim, expected_lags_len)
    
    # Check autocorrelation at lag 0 is 1.0
    assert np.allclose(autos[:, 0], 1.0)

def test_effective_sample_size():
    """Test effective_sample_size function"""
    # Create synthetic autocorrelation data that decays exponentially
    test_autocorr = np.exp(-np.arange(20)/5)
    ess = effective_sample_size(test_autocorr)
    
    # ESS should be positive
    assert ess > 0
    
    # For exponentially decaying autocorrelation, ESS should be less than N
    assert ess < len(test_autocorr)

def test_plot_lag(sample_data, img_kwargs):
    """Test plot_lag function with (dim, N) format data"""
    labels = ["Parameter 1", "Parameter 2", "Parameter 3"]
    maxlag = 50
    step = 5
    
    fig, axs = plot_lag(
        samples=sample_data,
        labels=labels,
        maxlag=maxlag,
        step=step,
        img_kwargs=img_kwargs
    )
    
    # Check return types
    assert isinstance(fig, Figure)
    assert isinstance(axs, Axes)
    
    # Test customization
    axs.set_title("Custom Lag Plot")
    test_filename = "test_lag_plot.png"
    fig.savefig(test_filename)
    
    # Verify file was created
    assert os.path.exists(test_filename)
    
    # Clean up
    plt.close(fig)
    os.remove(test_filename)

def test_1d_data_handling():
    """Test if functions handle 1D data correctly"""
    np.random.seed(42)
    N = 1000
    samples_1d = np.random.randn(1, N)
    
    # Test autocorrelation with 1D data
    lags, autos = autocorrelation(samples_1d, maxlag=50, step=5)
    assert autos.shape[0] == 1  # First dimension should be 1
    
    # Test plot_trace with 1D data
    fig, axs = plot_trace(samples_1d, labels=["Parameter 1"])
    plt.close(fig)
    
    # Test plot_lag with 1D data
    fig, axs = plot_lag(samples_1d, labels=["Parameter 1"], maxlag=50, step=5)
    plt.close(fig)

def test_dimension_validation():
    """Test that functions correctly validate input dimensions"""
    # Create data in wrong format (N, dim) instead of (dim, N)
    np.random.seed(42)
    N = 1000
    dim = 3
    wrong_format = np.random.randn(N, dim)  # (N, dim) format
    
    # This should raise a ValueError in all functions
    with pytest.raises(ValueError):
        autocorrelation(wrong_format)
    
    with pytest.raises(ValueError):
        plot_trace(wrong_format)
    
    with pytest.raises(ValueError):
        plot_lag(wrong_format, labels=["P1", "P2", "P3"])
