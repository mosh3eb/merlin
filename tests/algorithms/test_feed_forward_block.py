# MIT License
#
# Copyright (c)
#
# Tests for FeedForwardBlock API.

import math
from collections import defaultdict

import perceval as pcvl
import torch
from perceval.algorithm import Sampler
from perceval.components import PERM

from merlin.algorithms.feed_forward import FeedForwardBlock
from merlin.algorithms.layer import QuantumLayer
from merlin.core.computation_space import ComputationSpace
from merlin.measurement.strategies import MeasurementStrategy

_BASIS_CACHE: dict[tuple[int, int], list[tuple[int, ...]]] = {}


def _basis_states(n_modes: int, n_photons: int) -> list[tuple[int, ...]]:
    cache_key = (n_modes, n_photons)
    if cache_key not in _BASIS_CACHE:
        layer = QuantumLayer(
            input_size=0,
            circuit=pcvl.Circuit(n_modes),
            n_photons=n_photons,
            computation_space=ComputationSpace.FOCK,
        )
        _BASIS_CACHE[cache_key] = layer.computation_process.simulation_graph.mapped_keys
    return _BASIS_CACHE[cache_key]


def _as_keyed_tensors(block: FeedForwardBlock, tensor: torch.Tensor):
    keys = block.output_keys
    mapped: dict[tuple[int, ...], torch.Tensor] = {}
    for idx, key in enumerate(keys):
        entry = tensor[:, idx]
        if entry.shape[0] == 1:
            entry = entry.squeeze(0)
        try:
            size = block.output_state_sizes[key]
            if entry.shape[-1] > size:
                entry = entry[..., :size]
        except (AttributeError, KeyError, RuntimeError):
            pass
        mapped[key] = entry
    return mapped


def _build_balanced_feedforward_experiment():
    """Construct a small experiment with one detector and a feed-forward provider."""
    exp = pcvl.Experiment()
    root = pcvl.Circuit(3)
    root.add(0, pcvl.BS())
    exp.add(0, root)

    exp.add(0, pcvl.Detector.pnr())

    reflective = pcvl.Circuit(2)
    reflective.add(0, PERM([1, 0]))

    transmissive = pcvl.Circuit(2)
    transmissive.add(0, pcvl.BS())

    provider = pcvl.FFCircuitProvider(1, 0, reflective)
    provider.add_configuration([1], transmissive)
    exp.add(0, provider)
    return exp


def _build_two_stage_experiment():
    """Approximate the multi-level experiment from ff_perceval.py."""
    exp = pcvl.Experiment()
    root = pcvl.Circuit(4)
    root.add(0, pcvl.BS())
    exp.add(0, root)

    exp.add(0, pcvl.Detector.pnr())
    v0 = pcvl.Circuit(3) // pcvl.BS()
    v1 = pcvl.Circuit(3) // pcvl.BS()
    v2 = pcvl.Circuit(3) // pcvl.BS()
    provider1 = pcvl.FFCircuitProvider(1, 0, v0)
    provider1.add_configuration([1], v1)
    provider1.add_configuration([2], v2)
    exp.add(0, provider1)

    exp.add(3, pcvl.Detector.threshold())
    provider2 = pcvl.FFCircuitProvider(1, -1, pcvl.Circuit(2))
    provider2.add_configuration([1], pcvl.Circuit(2) // pcvl.BS())
    exp.add(3, provider2)

    for mode in (1, 2):
        exp.add(mode, pcvl.Detector.pnr())

    return exp


def test_feedforward_block2_balanced_split():
    exp = _build_balanced_feedforward_experiment()
    block = FeedForwardBlock(
        exp,
        input_state=[2, 0, 0],
    )

    x = torch.zeros((1, 0))
    outputs = block(x)
    distribution_map = _as_keyed_tensors(block, outputs)

    total_prob = 0.0
    measurement_probs = {}
    for key, distribution in distribution_map.items():
        prob = distribution.sum()
        total_prob += prob.item()
        measured_value = next(v for v in key if v is not None)
        measurement_probs[measured_value] = prob.item()

    assert math.isclose(total_prob, 1.0, rel_tol=1e-5)
    assert len(measurement_probs) == 3


def test_feedforward_block2_parses_multiple_stages():
    exp = _build_two_stage_experiment()
    block = FeedForwardBlock(exp, input_state=[1, 1, 0, 0])

    assert len(block.stages) == 2
    assert block.stages[0].measured_modes == (0,)
    assert block.stages[1].measured_modes == (3,)
    desc = block.describe()
    assert "Stage 1" in desc and "Stage 2" in desc


def test_feedforward_block2_matches_perceval_two_stage():
    exp = _build_two_stage_experiment()
    input_state = [1, 1, 0, 0]
    block = FeedForwardBlock(exp, input_state=input_state)

    x = torch.zeros((1, 0))
    block_outputs = block(x)
    distribution_map = _as_keyed_tensors(block, block_outputs)
    sample_key = block.output_keys[0]
    total_photons = sum(input_state)
    unmeasured_indices = [idx for idx, value in enumerate(sample_key) if value is None]
    final_runtime = block._stage_runtimes[-1]
    block_probs: defaultdict[tuple[int, ...], float] = defaultdict(float)
    for key, tensor in distribution_map.items():
        measured_sum = sum(v for v in key if v is not None)
        remaining_n = total_photons - measured_sum
        stage_key = tuple(key[idx] for idx in final_runtime.active_modes)
        reduced_key = tuple(
            0 if stage_key[idx] is None else int(stage_key[idx])
            for idx in final_runtime.measured_modes
        )
        layer = block._select_conditional_layer(
            final_runtime,
            reduced_key,
            remaining_n,
        )
        if layer is None:
            basis = _basis_states(len(unmeasured_indices), remaining_n)
        else:
            basis = layer.computation_process.simulation_graph.mapped_keys
        probabilities = tensor.reshape(-1).tolist()
        for state, prob in zip(basis, probabilities, strict=False):
            full_key = list(key)
            for mode_idx, value in zip(unmeasured_indices, state, strict=False):
                full_key[mode_idx] = value
            if prob > 1e-10:
                block_probs[tuple(full_key)] += prob

    exp.with_input(pcvl.BasicState(input_state))
    processor = pcvl.Processor("SLOS", exp)
    sampler = Sampler(processor)
    perceval_results = dict(sampler.probs()["results"])
    perceval_probs = {
        tuple(int(v) for v in state): float(prob)
        for state, prob in perceval_results.items()
    }
    assert set(block_probs) == set(perceval_probs)
    for key, value in block_probs.items():
        assert math.isclose(value, perceval_probs[key], rel_tol=1e-5, abs_tol=1e-5)


def test_feedforward_block2_amplitude_strategy_matches_probabilities():
    exp = _build_balanced_feedforward_experiment()
    input_state = [2, 0, 0]
    block_prob = FeedForwardBlock(exp, input_state=input_state)
    block_amp = FeedForwardBlock(
        exp,
        input_state=input_state,
        measurement_strategy=MeasurementStrategy.AMPLITUDES,
    )

    x = torch.zeros((1, 0))
    prob_outputs = block_prob(x)
    prob_map = _as_keyed_tensors(block_prob, prob_outputs)
    amp_outputs = block_amp(x)
    assert isinstance(amp_outputs, list)

    amp_prob_dict: dict[tuple[int, ...], torch.Tensor] = {}
    for measurement_key, branch_prob, _remaining_n, amp_tensor in amp_outputs:
        assert torch.is_complex(amp_tensor)
        prob = branch_prob
        while prob.ndim < amp_tensor.ndim:
            prob = prob.unsqueeze(-1)
        distribution = amp_tensor.abs().pow(2) * prob
        amp_prob_dict[measurement_key] = distribution

    for key, prob_tensor in prob_map.items():
        reconstructed = amp_prob_dict[key]
        assert torch.allclose(reconstructed, prob_tensor, atol=1e-6, rtol=1e-6), (
            f"Mismatch for key {key}"
        )


def test_feedforward_block2_mode_expectations():
    exp = _build_balanced_feedforward_experiment()
    input_state = [2, 0, 0]
    block_prob = FeedForwardBlock(exp, input_state=input_state)
    block_expect = FeedForwardBlock(
        exp,
        input_state=input_state,
        measurement_strategy=MeasurementStrategy.MODE_EXPECTATIONS,
    )

    x = torch.zeros((1, 0))
    prob_outputs = block_prob(x)
    expect_outputs = block_expect(x)
    prob_map = _as_keyed_tensors(block_prob, prob_outputs)
    expect_map = _as_keyed_tensors(block_expect, expect_outputs)
    total_photons = sum(input_state)
    sample_key = block_prob.output_keys[0]
    unmeasured_indices = [idx for idx, value in enumerate(sample_key) if value is None]
    for key, expectation in expect_map.items():
        measured_sum = sum(v for v in key if v is not None)
        remaining_n = total_photons - measured_sum
        basis = _basis_states(len(unmeasured_indices), remaining_n)
        distribution = prob_map[key].to(expectation.dtype)
        if basis:
            basis_tensor = torch.tensor(
                basis, dtype=expectation.dtype, device=expectation.device
            )
            manual = (distribution.unsqueeze(-1) * basis_tensor).sum(dim=-2)
        else:
            manual = torch.zeros_like(expectation)
        assert torch.allclose(manual, expectation, atol=1e-6, rtol=1e-6)


def test_feedforward_block2_accepts_tensor_input_state():
    exp = _build_balanced_feedforward_experiment()
    block_basic = FeedForwardBlock(exp, input_state=[2, 0, 0])
    basis = _basis_states(3, 2)
    amplitudes = torch.zeros(len(basis), dtype=torch.complex64)
    amplitudes[basis.index((2, 0, 0))] = 1.0
    block_tensor = FeedForwardBlock(exp, input_state=amplitudes)

    x = torch.zeros((1, 0))
    ref_outputs = _as_keyed_tensors(block_basic, block_basic(x))
    tensor_outputs = _as_keyed_tensors(block_tensor, block_tensor(x))
    for key in block_basic.output_keys:
        assert torch.allclose(ref_outputs[key], tensor_outputs[key], atol=1e-6)


def test_feedforward_block2_accepts_state_vector_input():
    exp = _build_balanced_feedforward_experiment()
    block_basic = FeedForwardBlock(exp, input_state=[2, 0, 0])
    state_vector = pcvl.StateVector()
    state_vector += pcvl.StateVector(pcvl.BasicState([2, 0, 0])) * 1.0
    block_sv = FeedForwardBlock(exp, input_state=state_vector)

    x = torch.zeros((1, 0))
    ref_outputs = _as_keyed_tensors(block_basic, block_basic(x))
    sv_outputs = _as_keyed_tensors(block_sv, block_sv(x))
    for key in block_basic.output_keys:
        assert torch.allclose(ref_outputs[key], sv_outputs[key], atol=1e-6)


def test_feedforward_block2_input_and_trainable_parameters_backward():
    exp = pcvl.Experiment()
    root = pcvl.Circuit(2)
    root.add(0, pcvl.PS(pcvl.P("phi")))
    root.add((0, 1), pcvl.BS(theta=pcvl.P("theta")))
    exp.add(0, root)
    exp.add(0, pcvl.Detector.pnr())

    conditional = pcvl.Circuit(1)
    conditional.add(0, pcvl.PS(pcvl.P("theta")))
    provider = pcvl.FFCircuitProvider(1, 0, conditional)
    exp.add(0, provider)

    block = FeedForwardBlock(
        exp,
        input_state=[1, 0],
        input_parameters=["phi"],
        trainable_parameters=["theta"],
    )

    x = torch.tensor([[0.1]], dtype=torch.float32, requires_grad=True)
    outputs = block(x)
    target_index = next(idx for idx, key in enumerate(block.output_keys) if key[0] == 1)
    loss = outputs[:, target_index].real.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.any(x.grad.abs() > 0)
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in block.parameters()
    )
