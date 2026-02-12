import numpy as np
import pytest

from samosa.core.model import ModelProtocol, validate_model_output


def test_validate_model_output_accepts_log_posterior():
    output = {"log_posterior": -1.23, "qoi": np.array([[1.0]])}
    validate_model_output(output)


def test_validate_model_output_accepts_log_components():
    output = {"log_prior": -0.5, "log_likelihood": -0.7, "cost": 2.0}
    validate_model_output(output)


def test_validate_model_output_rejects_non_dict():
    with pytest.raises(TypeError):
        validate_model_output([("log_posterior", -1.0)])  # type: ignore[arg-type]


def test_validate_model_output_rejects_missing_posterior_info():
    with pytest.raises(ValueError):
        validate_model_output({"qoi": np.array([[1.0]])})


def test_validate_model_output_rejects_partial_components_log_prior_only():
    with pytest.raises(ValueError):
        validate_model_output({"log_prior": -1.0})


def test_validate_model_output_rejects_partial_components_log_likelihood_only():
    with pytest.raises(ValueError):
        validate_model_output({"log_likelihood": -1.0})


def test_validate_model_output_rejects_mixed_direct_and_components():
    with pytest.raises(ValueError):
        validate_model_output(
            {"log_posterior": -1.0, "log_prior": -0.4, "log_likelihood": -0.6}
        )


class SimpleModel:
    def __call__(self, params: np.ndarray):
        return {"log_posterior": -0.5 * float(np.sum(params**2))}


class ComponentModel:
    def __call__(self, params: np.ndarray):
        return {
            "log_prior": -0.5 * float(np.sum(params**2)),
            "log_likelihood": -float(np.sum((params - 1.0) ** 2)),
        }


def test_model_protocol_runtime_check_for_callable_classes():
    assert isinstance(SimpleModel(), ModelProtocol)
    assert isinstance(ComponentModel(), ModelProtocol)
