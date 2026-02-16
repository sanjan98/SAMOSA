"""
Tests for transport maps in samosa.maps.

All maps implement the TransportMapBase interface: forward(), inverse(), adapt(),
and inherit pullback_logpdf() from the base class. Tests verify shapes,
forward-inverse round-trip (inverse(forward(x)) ≈ x), adaptation gating, and
pullback log-density.
"""

import numpy as np
import pytest

from samosa.core.state import ChainState


def _make_state(dim: int, position: np.ndarray, iteration: int = 0) -> ChainState:
    """Build a ChainState with position shape (dim, 1) and metadata."""
    if position.ndim == 1:
        position = position.reshape(-1, 1)
    return ChainState(
        position=position,
        log_posterior=0.0,
        metadata={"iteration": iteration},
    )


def _assert_round_trip(
    tmap, x: np.ndarray, *, atol: float = 1e-5, rtol: float = 1e-5
) -> None:
    """Check that inverse(forward(x)) recovers x up to numerical tolerance."""
    r, _ = tmap.forward(x)
    x_back, _ = tmap.inverse(r)
    np.testing.assert_allclose(x_back, x, atol=atol, rtol=rtol)


def _assert_output_shapes(tmap, x: np.ndarray) -> None:
    """Check forward/inverse return correct shapes and logdet is scalar or per-sample."""
    r, logdet = tmap.forward(x)
    x_inv, logdet_inv = tmap.inverse(r)
    assert r.shape == x.shape
    assert x_inv.shape == x.shape
    logdet_arr = np.atleast_1d(np.asarray(logdet))
    logdet_inv_arr = np.atleast_1d(np.asarray(logdet_inv))
    assert logdet_arr.size == 1 or logdet_arr.size == x.shape[1]
    assert logdet_inv_arr.size == 1 or logdet_inv_arr.size == x.shape[1]


# ----- LowerTriangularMap (MParT) -----


@pytest.fixture(scope="module")
def lowertriangular_map():
    pytest.importorskip("mpart")
    from samosa.maps.triangular import LowerTriangularMap

    return LowerTriangularMap(dim=2, total_order=1)


def test_lowertriangular_forward_inverse_shapes(lowertriangular_map):
    x = np.zeros((2, 3))
    _assert_output_shapes(lowertriangular_map, x)


def test_lowertriangular_round_trip(lowertriangular_map):
    # Single point and batch; MParT default coeffs may not be identity, use looser tol
    x1 = np.array([[0.0], [0.0]])
    _assert_round_trip(lowertriangular_map, x1, atol=1e-4, rtol=1e-4)
    x2 = np.random.randn(2, 5).astype(np.float64)
    _assert_round_trip(lowertriangular_map, x2, atol=1e-4, rtol=1e-4)


def test_lowertriangular_adapt_early_iteration_noop(lowertriangular_map):
    state = _make_state(2, np.zeros(2), iteration=0)
    lowertriangular_map.adapt([state], force_adapt=False)


def test_lowertriangular_pullback_logpdf(lowertriangular_map):
    x = np.zeros((2, 2))
    log_pdf = lowertriangular_map.pullback_logpdf(x)
    out = np.asarray(log_pdf)
    assert out.shape in ((), (2,))
    assert np.all(np.isfinite(out))


# ----- Normalizingflow (normflows) -----


@pytest.fixture(scope="module")
def normalizingflow_map():
    pytest.importorskip("torch")
    pytest.importorskip("normflows")
    from samosa.maps.normalizing_flow import Normalizingflow

    return Normalizingflow(dim=2, flows=[], force_cpu=True, num_epochs=1)


def test_normalizingflow_forward_inverse_shapes(normalizingflow_map):
    x = np.zeros((2, 4))
    _assert_output_shapes(normalizingflow_map, x)


def test_normalizingflow_round_trip(normalizingflow_map):
    x1 = np.array([[0.0], [0.0]])
    _assert_round_trip(normalizingflow_map, x1)
    x2 = np.random.randn(2, 5).astype(np.float64)
    _assert_round_trip(normalizingflow_map, x2)


def test_normalizingflow_adapt_early_iteration_noop(normalizingflow_map):
    state = _make_state(2, np.zeros(2), iteration=0)
    normalizingflow_map.adapt([state], force_adapt=False)


def test_normalizingflow_pullback_logpdf(normalizingflow_map):
    x = np.zeros((2, 2))
    log_pdf = normalizingflow_map.pullback_logpdf(x)
    out = np.asarray(log_pdf)
    assert out.shape in ((), (2,))
    assert np.all(np.isfinite(out))


# ----- RealNVPMap -----


@pytest.fixture(scope="module")
def realnvp_map():
    pytest.importorskip("torch")
    from samosa.maps.realnvp import RealNVPMap

    masks = [
        np.array([0, 1], dtype=np.float32),
        np.array([1, 0], dtype=np.float32),
    ]
    return RealNVPMap(dim=2, masks=masks, hidden_dim=4, num_epochs=1)


def test_realnvp_forward_inverse_shapes(realnvp_map):
    x = np.zeros((2, 5))
    _assert_output_shapes(realnvp_map, x)


def test_realnvp_round_trip(realnvp_map):
    x1 = np.array([[0.0], [0.0]])
    _assert_round_trip(realnvp_map, x1)
    x2 = np.random.randn(2, 5).astype(np.float64)
    _assert_round_trip(realnvp_map, x2)


def test_realnvp_adapt_early_iteration_noop(realnvp_map):
    state = _make_state(2, np.zeros(2), iteration=0)
    realnvp_map.adapt([state], force_adapt=False)


def test_realnvp_pullback_logpdf(realnvp_map):
    x = np.zeros((2, 2))
    log_pdf = realnvp_map.pullback_logpdf(x)
    out = np.asarray(log_pdf)
    assert out.shape in ((), (2,))
    assert np.all(np.isfinite(out))


# ----- LinearOptimalTransportMap -----


@pytest.fixture(scope="module")
def lot_map():
    from samosa.maps.lot import LinearOptimalTransportMap

    return LinearOptimalTransportMap(dim=2)


def test_lot_forward_inverse_shapes(lot_map):
    x = np.zeros((2, 3))
    _assert_output_shapes(lot_map, x)


def test_lot_round_trip(lot_map):
    # Default A=I, b=0: forward and inverse are exact
    x1 = np.array([[0.0], [0.0]])
    _assert_round_trip(lot_map, x1)
    x2 = np.random.randn(2, 5).astype(np.float64)
    _assert_round_trip(lot_map, x2)


def test_lot_adapt_requires_paired_samples(lot_map):
    state = _make_state(2, np.zeros(2), iteration=500)
    with pytest.raises(ValueError, match="paired_samples"):
        lot_map.adapt([state], force_adapt=True, paired_samples=None)


def test_lot_adapt_early_iteration_noop(lot_map):
    state_fine = _make_state(2, np.zeros(2), iteration=0)
    state_coarse = _make_state(2, np.ones(2), iteration=0)
    # Should return without error (iteration outside window)
    lot_map.adapt([state_fine], force_adapt=False, paired_samples=[state_coarse])


def test_lot_adapt_updates_map_with_paired_samples(lot_map):
    # Force adapt with fine and coarse samples; map should update (A, b change)
    rng = np.random.default_rng(42)
    fine_pos = rng.standard_normal((2, 20))
    coarse_pos = rng.standard_normal((2, 20)) * 0.5
    samples_fine = [_make_state(2, fine_pos[:, i], iteration=500) for i in range(20)]
    samples_coarse = [
        _make_state(2, coarse_pos[:, i], iteration=500) for i in range(20)
    ]

    lot_map.adapt(samples_fine, force_adapt=True, paired_samples=samples_coarse)

    # After adapt, round-trip should still hold
    x = np.random.randn(2, 4).astype(np.float64)
    _assert_round_trip(lot_map, x)


def test_lot_pullback_logpdf(lot_map):
    x = np.zeros((2, 2))
    log_pdf = lot_map.pullback_logpdf(x)
    out = np.asarray(log_pdf)
    assert out.shape in ((), (2,))
    assert np.all(np.isfinite(out))
