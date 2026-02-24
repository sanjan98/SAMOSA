"""
Unit tests for core kernel utilities used by single-chain (MH/DR).
"""

import numpy as np
import pytest

from samosa.core.kernel import _marginal_mh_acceptance_ratio


def test_marginal_mh_acceptance_ratio_better_proposal():
    """Better proposal (higher log posterior) with symmetric q => 1.0."""
    lpc, lpp = -2.0, -0.5
    logq_fwd, logq_rev = 0.0, 0.0
    ar = _marginal_mh_acceptance_ratio(lpc, lpp, logq_fwd, logq_rev)
    assert ar == 1.0


def test_marginal_mh_acceptance_ratio_worse_proposal():
    """Worse proposal => value in (0, 1), exp(check)."""
    lpc, lpp = -0.5, -2.0
    logq_fwd, logq_rev = 0.0, 0.0
    ar = _marginal_mh_acceptance_ratio(lpc, lpp, logq_fwd, logq_rev)
    assert 0 < ar < 1
    expected = np.exp((lpp + logq_rev) - (lpc + logq_fwd))
    assert np.isclose(ar, expected)


def test_marginal_mh_acceptance_ratio_asymmetric_q():
    """Asymmetric q: forward and reverse logq differ."""
    lpc, lpp = -1.0, -0.5
    logq_fwd, logq_rev = -0.2, -0.3
    ar = _marginal_mh_acceptance_ratio(lpc, lpp, logq_fwd, logq_rev)
    check = (lpp + logq_rev) - (lpc + logq_fwd)
    expected = 1.0 if check > 0 else float(np.exp(check))
    assert np.isclose(ar, expected)
    assert 0 <= ar <= 1


def test_marginal_mh_acceptance_ratio_proposed_inf():
    """Proposed log_posterior = -inf => ratio 0."""
    lpc, lpp = -1.0, -np.inf
    logq_fwd, logq_rev = 0.0, 0.0
    ar = _marginal_mh_acceptance_ratio(lpc, lpp, logq_fwd, logq_rev)
    assert ar == 0.0


def test_marginal_mh_acceptance_ratio_current_inf():
    """Current log_posterior = -inf, proposed finite => accept (1.0 or exp)."""
    lpc, lpp = -np.inf, -1.0
    logq_fwd, logq_rev = 0.0, 0.0
    ar = _marginal_mh_acceptance_ratio(lpc, lpp, logq_fwd, logq_rev)
    assert ar == 1.0


def test_marginal_mh_acceptance_ratio_both_finite():
    """Both finite: ratio is min(1, exp(check))."""
    lpc, lpp = -3.0, -2.0
    logq_fwd, logq_rev = -0.1, -0.1
    ar = _marginal_mh_acceptance_ratio(lpc, lpp, logq_fwd, logq_rev)
    assert np.isfinite(ar)
    assert 0 <= ar <= 1
