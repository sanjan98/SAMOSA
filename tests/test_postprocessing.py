import os
import pickle
import tempfile

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

from samosa.core.state import ChainState
from samosa.utils.post_processing import (
    autocorrelation,
    effective_sample_size,
    extract_from_states,
    get_position_from_states,
    get_reference_position_from_states,
    joint_plots,
    load_coupled_samples,
    load_samples,
    plot_lag,
    plot_trace,
    scatter_matrix,
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
        "label_fontsize": 24,
        "title_fontsize": 20,
        "tick_fontsize": 20,
        "legend_fontsize": 16,
        "img_format": "png",
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
    fig.savefig(test_filename, bbox_inches="tight")

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
    axs_list = axs if isinstance(axs, list) else [axs]
    axs_list[0].set_title("Custom Title")

    test_filename = "test_trace_plot.png"
    fig.savefig(test_filename)

    # Verify file was created
    assert os.path.exists(test_filename)

    # Clean up
    plt.close(fig)
    os.remove(test_filename)


def test_autocorrelation(sample_data):
    """Test autocorrelation function with (dim, N) format data"""
    autos, taus, ess = autocorrelation(sample_data, c=5, tol=50)

    assert isinstance(autos, np.ndarray)
    assert isinstance(taus, np.ndarray)
    assert isinstance(ess, np.ndarray)

    dim, nsamples = sample_data.shape
    assert autos.shape == (dim, nsamples)
    assert taus.shape == (dim,)
    assert ess.shape == (dim,)

    # Autocorrelation at lag 0 is 1.0
    assert np.allclose(autos[:, 0], 1.0)


def test_effective_sample_size():
    """Test effective_sample_size function"""
    # Create synthetic autocorrelation data that decays exponentially
    test_autocorr = np.exp(-np.arange(20) / 5)
    ess = effective_sample_size(test_autocorr)

    # ESS should be positive
    assert ess > 0

    # For exponentially decaying autocorrelation, ESS should be less than N
    assert ess < len(test_autocorr)


def test_plot_lag(sample_data, img_kwargs):
    """Test plot_lag function with (dim, N) format data"""
    labels = ["Parameter 1", "Parameter 2", "Parameter 3"]
    fig, axs, all_ess, all_taus = plot_lag(
        samples=sample_data, labels=labels, maxlag=50, step=5, img_kwargs=img_kwargs
    )

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
    autos, taus, ess = autocorrelation(samples_1d, c=5, tol=50)
    assert autos.shape[0] == 1

    # Test plot_trace with 1D data
    fig, axs = plot_trace(samples_1d, labels=["Parameter 1"])
    plt.close(fig)

    # Test plot_lag with 1D data
    fig, axs, _, _ = plot_lag(samples_1d, labels=["Parameter 1"], maxlag=50, step=5)
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


# --- Tests for load_samples, load_coupled_samples ---


@pytest.fixture
def chain_states():
    """Create a list of ChainState objects for testing."""
    np.random.seed(42)
    dim = 2
    N = 50
    positions = np.random.randn(dim, N)
    return [
        ChainState(
            position=positions[:, i : i + 1],
            log_posterior=float(-0.5 * np.sum(positions[:, i] ** 2)),
            metadata={"iteration": i + 1},
        )
        for i in range(N)
    ]


def test_load_samples(chain_states):
    """Test load_samples loads from pickle file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "samples.pkl")
        with open(path, "wb") as f:
            pickle.dump(chain_states, f)
        loaded = load_samples(tmpdir)
        assert len(loaded) == len(chain_states)
        assert all(isinstance(s, ChainState) for s in loaded)
        np.testing.assert_array_almost_equal(
            loaded[0].position, chain_states[0].position
        )


def test_load_samples_iteration(chain_states):
    """Test load_samples with iteration (checkpoint) path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "samples"), exist_ok=True)
        path = os.path.join(tmpdir, "samples", "samples_100.pkl")
        with open(path, "wb") as f:
            pickle.dump(chain_states, f)
        loaded = load_samples(tmpdir, iteration=100)
        assert len(loaded) == len(chain_states)


def test_load_samples_file_not_found():
    """Test load_samples raises when file does not exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            load_samples(tmpdir)
        with pytest.raises(FileNotFoundError):
            load_samples(tmpdir, iteration=999)


def test_load_coupled_samples(chain_states):
    """Test load_coupled_samples loads coarse and fine from pickle files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "samples_coarse.pkl"), "wb") as f:
            pickle.dump(chain_states, f)
        with open(os.path.join(tmpdir, "samples_fine.pkl"), "wb") as f:
            pickle.dump(chain_states[:30], f)
        coarse, fine = load_coupled_samples(tmpdir)
        assert len(coarse) == len(chain_states)
        assert len(fine) == 30


def test_load_coupled_samples_iteration(chain_states):
    """Test load_coupled_samples with iteration (checkpoint) path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "samples"), exist_ok=True)
        with open(os.path.join(tmpdir, "samples", "samples_coarse_50.pkl"), "wb") as f:
            pickle.dump(chain_states, f)
        with open(os.path.join(tmpdir, "samples", "samples_fine_50.pkl"), "wb") as f:
            pickle.dump(chain_states, f)
        coarse, fine = load_coupled_samples(tmpdir, iteration=50)
        assert len(coarse) == len(chain_states)


# --- Tests for extract_from_states, get_position_from_states ---


def test_extract_from_states_position(chain_states):
    """Test extract_from_states with attribute='position'."""
    out = extract_from_states(chain_states, attribute="position", burnin=0.1)
    assert out.shape[0] == 2
    n_after_burnin = int(len(chain_states) * 0.9)
    assert out.shape[1] == n_after_burnin


def test_extract_from_states_log_posterior(chain_states):
    """Test extract_from_states with attribute='log_posterior'."""
    out = extract_from_states(chain_states, attribute="log_posterior")
    assert out.shape == (1, len(chain_states))
    assert np.all(np.isfinite(out))


def test_get_position_from_states(chain_states):
    """Test get_position_from_states convenience wrapper."""
    out = get_position_from_states(chain_states, burnin=0.2)
    assert out.shape[0] == 2
    assert out.shape[1] == int(len(chain_states) * 0.8)


def test_get_reference_position_from_states():
    """Test get_reference_position_from_states with states that have reference_position."""
    np.random.seed(42)
    states = [
        ChainState(
            position=np.array([[1.0], [2.0]]),
            reference_position=np.array([[0.5], [1.0]]),
            log_posterior=-3.0,
        )
        for _ in range(10)
    ]
    out = get_reference_position_from_states(states)
    assert out.shape == (2, 10)
    np.testing.assert_array_almost_equal(out[:, 0], [0.5, 1.0])


def test_extract_from_states_reference_position_none_raises():
    """Test extract_from_states raises when reference_position is None."""
    states = [
        ChainState(position=np.array([[1.0], [2.0]]), log_posterior=-3.0)
        for _ in range(5)
    ]
    with pytest.raises(ValueError, match="reference_position"):
        extract_from_states(states, attribute="reference_position")


# --- Tests for joint_plots ---


def test_joint_plots(multiple_samples):
    """Test joint_plots with two sample chains."""
    figures = joint_plots(multiple_samples, bins=20)
    assert len(figures) == multiple_samples[0].shape[0]
    assert all(isinstance(f, Figure) for f in figures)
    for f in figures:
        plt.close(f)


def test_joint_plots_requires_two_samples():
    """Test joint_plots raises when not exactly 2 samples."""
    with pytest.raises(ValueError, match="Need 2 samples"):
        joint_plots([np.random.randn(2, 100)])
    with pytest.raises(ValueError, match="Need 2 samples"):
        joint_plots(
            [
                np.random.randn(2, 100),
                np.random.randn(2, 100),
                np.random.randn(2, 100),
            ]
        )
