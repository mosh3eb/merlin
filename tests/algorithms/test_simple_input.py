from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from merlin import MeasurementDistribution, QuantumLayer
from merlin.utils.grouping.mappers import ModGrouping

_PCVL_HOME = Path(__file__).resolve().parents[2] / ".pcvl_home"
(
    _PCVL_HOME / "Library" / "Application Support" / "perceval-quandela" / "job_group"
).mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(_PCVL_HOME)


@pytest.fixture(autouse=True)
def perceval_home(monkeypatch):
    monkeypatch.setenv("HOME", str(_PCVL_HOME))


@pytest.fixture
def quantum_layer_api():
    return QuantumLayer


def _unwrap(layer: nn.Module) -> QuantumLayer:
    return layer[0] if isinstance(layer, nn.Sequential) else layer


def test_none_strategy_without_output_size(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    layer = QuantumLayer.simple(
        input_size=3,
        n_params=60,
        dtype=torch.float32,
    )
    base = _unwrap(layer)

    x = torch.rand(4, 3)
    output = layer(x)

    assert output.shape == (4, base.output_size)
    assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-5)


def test_none_strategy_with_matching_output_size(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    reference_layer = QuantumLayer.simple(
        input_size=3,
        n_params=60,
    )
    dist_size = _unwrap(reference_layer).output_size

    layer = QuantumLayer.simple(
        input_size=3,
        n_params=60,
        output_size=dist_size,
    )

    base = _unwrap(layer)
    x = torch.rand(2, 3)
    output = layer(x)
    assert output.shape == (2, dist_size)
    assert base.output_size == dist_size


def test_simple_groups_output_when_requested(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    target_size = 16
    layer = QuantumLayer.simple(
        input_size=3,
        n_params=40,
        output_size=target_size,
    )

    assert isinstance(layer, nn.Sequential)
    assert isinstance(layer[1], ModGrouping)
    base = _unwrap(layer)
    x = torch.rand(4, 3)
    output = layer(x)
    assert base.output_size != target_size
    assert output.shape == (4, target_size)
    assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-5)


def test_linear_strategy_creates_linear_mapping(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    layer = QuantumLayer.simple(
        input_size=3,
        n_params=60,
    )
    model = nn.Sequential(layer, nn.Linear(layer.output_size, 5))

    assert isinstance(layer.measurement_mapping, MeasurementDistribution)
    x = torch.rand(6, 3)
    output = model(x)
    assert output.shape == (6, 5)


def test_default_strategy_is_none(quantum_layer_api):
    QuantumLayer = quantum_layer_api
    sig = inspect.signature(QuantumLayer.simple)
    assert "measurement_strategy" not in sig.parameters


def test_trainable_parameter_budget_matches_request(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    requested_params = 37
    with pytest.warns(RuntimeWarning):
        layer = QuantumLayer.simple(
            input_size=3,
            n_params=requested_params,
        )

    theta_param_count = sum(
        param.numel()
        for name, param in layer.named_parameters()
        if name.startswith("theta")
    )
    interferometer_param_count = sum(
        param.numel()
        for name, param in layer.named_parameters()
        if name.startswith("gi_simple")
    )

    total_trainable = theta_param_count + interferometer_param_count
    expected_total = max(requested_params, interferometer_param_count)
    expected_theta = max(requested_params - interferometer_param_count, 0)

    assert total_trainable == expected_total
    assert theta_param_count == expected_theta


def test_gradient_flow_for_strategies(quantum_layer_api):
    QuantumLayer = quantum_layer_api
    nb_params = 40

    layer = QuantumLayer.simple(
        input_size=3,
        n_params=nb_params,
    )
    model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 4))

    x = torch.rand(8, 3, requires_grad=True)
    loss = model(x).sum()
    loss.backward()
    assert any(
        p.grad is not None and torch.any(p.grad != 0) for p in model.parameters()
    )

    layer_none = QuantumLayer.simple(
        input_size=3,
        n_params=nb_params,
    )

    x = torch.rand(8, 3, requires_grad=True)
    loss = layer_none(x).sum()
    loss.backward()
    assert any(
        p.grad is not None and torch.any(p.grad != 0) for p in layer_none.parameters()
    )
    theta_param_count = sum(
        param.numel()
        for name, param in layer_none.named_parameters()
        if name.startswith("theta")
    )
    interferometer_param_count = sum(
        param.numel()
        for name, param in layer_none.named_parameters()
        if name.startswith("gi_simple")
    )

    total_trainable = theta_param_count + interferometer_param_count
    expected_total = max(nb_params, interferometer_param_count)
    expected_theta = max(nb_params - interferometer_param_count, 0)

    assert total_trainable == expected_total
    assert theta_param_count == expected_theta


def test_quantum_layer_simple_raises_when_input_exceeds_modes(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    with pytest.raises(
        ValueError, match="You cannot encore more features than mode with Builder"
    ):
        QuantumLayer.simple(
            input_size=12,
            n_params=30,
        )


def test_batch_shapes_and_probabilities(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    layer = QuantumLayer.simple(
        input_size=4,
        n_params=80,
    )

    for batch_size in [1, 5, 16]:
        x = torch.rand(batch_size, 4)
        output = layer(x)
        assert output.shape == (batch_size, layer.output_size)
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        assert torch.all(output >= 0)


def test_dtype_propagation(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    for dtype in (torch.float32, torch.float64):
        layer = QuantumLayer.simple(
            input_size=3,
            n_params=60,
            dtype=dtype,
        )

        for param in layer.parameters():
            assert param.dtype == dtype

        x = torch.rand(2, 3, dtype=dtype)
        output = layer(x)
        assert output.dtype == dtype
