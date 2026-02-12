import numpy as np
import pytest

from samosa.core.state import ChainState


def test_state_accepts_column_vector_position():
    state = ChainState(position=np.array([[1.0], [2.0]]), log_posterior=-1.0)
    assert state.position.shape == (2, 1)


def test_state_rejects_1d_position():
    with pytest.raises(ValueError):
        ChainState(position=np.array([1.0, 2.0]), log_posterior=-1.0)


def test_state_rejects_non_array_position():
    with pytest.raises(TypeError):
        ChainState(position=[1.0, 2.0], log_posterior=-1.0)  # type: ignore[arg-type]


def test_state_allows_missing_metadata():
    state = ChainState(
        position=np.array([[1.0], [2.0]]), log_posterior=-1.0, metadata=None
    )
    assert state.metadata is None


def test_state_reference_position_shape_checks():
    with pytest.raises(ValueError):
        ChainState(
            position=np.array([[1.0], [2.0]]),
            reference_position=np.array([1.0, 2.0]),
            log_posterior=-1.0,
        )

    with pytest.raises(ValueError):
        ChainState(
            position=np.array([[1.0], [2.0]]),
            reference_position=np.array([[1.0], [2.0], [3.0]]),
            log_posterior=-1.0,
        )


def test_state_computes_posterior_from_components():
    state = ChainState(
        position=np.array([[1.0], [2.0]]),
        log_prior=-2.0,
        log_likelihood=-3.0,
    )
    assert state.log_posterior == -5.0
    assert state.posterior == -5.0


def test_state_posterior_raises_when_missing():
    state = ChainState(position=np.array([[1.0], [2.0]]))
    with pytest.raises(ValueError):
        _ = state.posterior


def test_state_validate_passes_with_log_posterior():
    state = ChainState(position=np.array([[1.0], [2.0]]), log_posterior=-4.0)
    state.validate()


def test_state_validate_passes_with_components():
    state = ChainState(
        position=np.array([[1.0], [2.0]]),
        log_prior=-2.0,
        log_likelihood=-3.0,
    )
    state.validate()


def test_state_validate_raises_when_posterior_undefined():
    state = ChainState(position=np.array([[1.0], [2.0]]))
    with pytest.raises(ValueError):
        state.validate()


def test_state_repr_handles_missing_posterior():
    state = ChainState(position=np.array([[1.0], [2.0]]))
    rep = repr(state)
    assert "posterior=?" in rep


def test_state_repr_includes_posterior_when_available():
    state = ChainState(position=np.array([[1.0], [2.0]]), log_posterior=-1.23456)
    rep = repr(state)
    assert "posterior=-1.2346" in rep
