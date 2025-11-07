from __future__ import annotations

import perceval as pcvl
import pytest
import torch

from merlin.measurement import DetectorTransform
from merlin.utils.combinadics import Combinadics


def test_partial_detector_transform_outputs_probability_and_amplitudes():
    simulation_keys = [(1, 0), (0, 1)]
    detectors = [pcvl.Detector.threshold(), None]
    transform = DetectorTransform(simulation_keys, detectors, partial_measurement=True)

    amplitudes = torch.tensor([1.0 + 0.0j, 0.0 + 0.0j])
    result = transform(amplitudes)

    assert set(result.keys()) == {(1,), (0,)}
    assert transform.output_keys == [(0,), (1,)]

    prob_one, nphotons_one, rem_one = result[(1,)]
    prob_zero, nphotons_zero, rem_zero = result[(0,)]

    assert torch.allclose(prob_one, torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(prob_zero, torch.tensor(0.0), atol=1e-6)
    assert rem_one.shape[-1] == len(transform.output_keys)
    assert rem_zero.shape[-1] == len(transform.output_keys)
    assert nphotons_one.item() == 0
    assert nphotons_zero.item() == 1


def test_partial_detector_transform_requires_complex_tensor():
    simulation_keys = [(1, 0), (0, 1)]
    detectors = [pcvl.Detector.threshold(), None]
    transform = DetectorTransform(simulation_keys, detectors, partial_measurement=True)

    with pytest.raises(TypeError, match="complex-valued"):
        transform(torch.ones(2))


def test_partial_detector_transform_preserves_batch_dimension():
    simulation_keys = [(1, 0), (0, 1)]
    detectors = [pcvl.Detector.threshold(), None]
    transform = DetectorTransform(simulation_keys, detectors, partial_measurement=True)

    amplitudes = torch.tensor([
        [1.0 + 0.0j, 0.0 + 0.0j],
        [0.0 + 0.0j, 1.0 + 0.0j],
    ])
    result = transform(amplitudes)

    assert set(result.keys()) == {(1,), (0,)}

    for probability, nphotons, collapsed in result.values():
        assert probability.shape == (amplitudes.shape[0],)
        assert collapsed.shape == amplitudes.shape
        assert nphotons.dim() == 0

    total = torch.zeros(amplitudes.shape[0], device=amplitudes.device)
    for probability, *_ in result.values():
        total = total + probability
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)


def test_partial_detector_transform_handles_no_detectors():
    simulation_keys = [(1, 0), (0, 1)]
    detectors = [None, None]
    transform = DetectorTransform(simulation_keys, detectors, partial_measurement=True)

    amplitudes = torch.tensor([1.0 + 0.0j, 0.0 + 0.0j])
    result = transform(amplitudes)

    assert set(result.keys()) == {()}
    probability, nphotons, collapsed = result[()]
    assert torch.allclose(probability, torch.tensor(1.0), atol=1e-6)
    assert nphotons.item() == 1
    assert torch.allclose(collapsed, amplitudes)


def test_partial_detector_transform_mixed_pnr_case():
    combinator = Combinadics("fock", n=3, m=4)
    detectors = [pcvl.Detector.pnr(), pcvl.Detector.pnr(), None, None]
    transform = DetectorTransform(
        combinator.iter_states(),
        detectors,
        partial_measurement=True,
    )

    basis_states = combinator.enumerate_states()
    amplitudes_list: list[complex] = []
    for idx, _state in enumerate(basis_states):
        real = 0.05 * (idx + 1)
        imag = ((-1) ** idx) * 0.03 * (idx + 2)
        amplitudes_list.append(complex(real, imag))

    amplitudes = torch.tensor(amplitudes_list, dtype=torch.complex64)

    result = transform(amplitudes)
    remaining_keys = transform.output_keys
    assert set(remaining_keys) == {
        (0, 3),
        (1, 2),
        (2, 1),
        (3, 0),
        (0, 2),
        (1, 1),
        (2, 0),
        (0, 1),
        (1, 0),
        (0, 0),
    }

    expected_outcomes = {(state[0], state[1]) for state in basis_states}
    assert set(result.keys()) == expected_outcomes

    for meas_key, (probability, nphotons, collapsed) in result.items():
        assert isinstance(nphotons, torch.Tensor)
        expected_vector = torch.zeros(len(remaining_keys), dtype=amplitudes.dtype)
        for key, amp in zip(basis_states, amplitudes, strict=False):
            if (key[0], key[1]) != meas_key:
                continue
            reduced = (key[2], key[3])
            idx = remaining_keys.index(reduced)
            expected_vector[idx] += amp
        assert torch.allclose(collapsed, expected_vector)
        expected_prob = expected_vector.abs().pow(2).sum()
        assert torch.allclose(probability, expected_prob, atol=1e-6)
        remaining_n_expected = 3 - sum(meas_key)
        assert nphotons.item() == remaining_n_expected
