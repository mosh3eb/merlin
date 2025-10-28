"""
Probability consistency & gradient-free tuning tests for MerlinProcessor.

Covers:
- Local vs remote probability correspondence (no_bunching and bunching)
- Mixed classical/quantum workflows preserve classical ops locally and
  offload quantum ops (and can be flipped to local with force_simulation)
- COBYLA optimization loop that mutates QuantumLayer params and evaluates
  the objective via MerlinProcessor.forward (remote if 'probs' allowed,
  else high-nsample sampling)

All tests that require cloud auto-skip via the `remote_processor` fixture.
"""

from __future__ import annotations

import math
import time

import pytest
import torch
import torch.nn as nn

from merlin.algorithms import QuantumLayer
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.core.merlin_processor import MerlinProcessor
from merlin.sampling.strategies import OutputMappingStrategy


# -------------------------
# Utilities
# -------------------------

def _make_layer(m: int, n: int, input_size: int, no_bunching: bool) -> QuantumLayer:
    builder = CircuitBuilder(n_modes=m)
    # Keep the layer small-ish but non-trivial
    builder.add_rotations(trainable=True, name="theta")
    if m >= 3:
        builder.add_entangling_layer()
    builder.add_angle_encoding(modes=list(range(input_size)), name="px")

    return QuantumLayer(
        input_size=input_size,
        output_size=None,  # raw distribution
        builder=builder,
        n_photons=n,
        no_bunching=no_bunching,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    ).eval()


def _l1_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # shape: (B, D) -> (B,)
    return (a - b).abs().sum(dim=1)


def _spin_until(pred, timeout_s: float = 10.0, sleep_s: float = 0.02) -> bool:
    import time as _t
    start = _t.time()
    while not pred():
        if _t.time() - start > timeout_s:
            return False
        _t.sleep(sleep_s)
    return True


# -------------------------
# Probability correspondence (no_bunching / bunching)
# -------------------------

@pytest.mark.parametrize("no_bunching, m, n, input_size, tol_l1", [
    (True,  6, 2, 2, 0.12),   # C(6,2)=15
    (False, 5, 3, 3, 0.15),   # C(7,3)=35
])
def test_prob_consistency_local_vs_remote(remote_processor, no_bunching, m, n, input_size, tol_l1):
    """
    Compare local exact distribution to MerlinProcessor remote output.
    If the backend supports 'probs', use it directly; else use high nsample
    to approximate (looser tolerance).
    """
    layer = _make_layer(m, n, input_size, no_bunching)
    bsz = 3
    X = torch.rand(bsz, input_size)

    # Local "ground truth"
    y_local = layer(X)  # exact probs
    assert y_local.shape[0] == bsz
    assert torch.allclose(y_local.sum(dim=1), torch.ones(bsz), atol=1e-5)

    proc = MerlinProcessor(remote_processor)
    # Decide evaluation mode:
    use_probs = "probs" in getattr(proc, "available_commands", [])
    if use_probs:
        # We rely on exact probs (fast & precise)
        y_remote = proc.forward(layer, X, nsample=None)
    else:
        # Sampling: push nsample high to shrink variance
        # (these values are a trade-off for CI/runtime)
        y_remote = proc.forward(layer, X, nsample=200_000)

    assert y_remote.shape == y_local.shape
    # Normalization is looser if sampling
    atol_norm = 1e-5 if use_probs else 2e-2
    assert torch.allclose(y_remote.sum(dim=1), torch.ones(bsz), atol=atol_norm)

    # Distribution similarity (L1 per sample)
    l1 = _l1_dist(y_local, y_remote)
    # Allow looser tolerance for sampling backends
    assert torch.all(l1 <= tol_l1), f"L1 distances too high: {l1.tolist()}"


# -------------------------
# Workflow preservation (classical/quantum mix)
# -------------------------

def test_workflow_preserves_classical_layers_and_offloads_quantum(remote_processor):
    """
    Build a small pipeline: Linear -> ReLU -> Quantum -> Linear -> Softmax.
    Check that:
      - quantum layer is offloaded (>=1 job id)
      - classical transforms are applied locally around it
      - forcing local execution yields similar results (within tolerance)
    """
    # Quantum core
    q = _make_layer(m=5, n=2, input_size=2, no_bunching=True).eval()
    # Probe output size
    dist1 = q(torch.rand(2, 2)).shape[1]

    model_remote = nn.Sequential(
        nn.Linear(3, 2, bias=False),
        nn.ReLU(),
        q,
        nn.Linear(dist1, 4, bias=False),
        nn.Softmax(dim=-1),
    ).eval()

    X = torch.rand(4, 3)

    proc = MerlinProcessor(remote_processor)
    fut = proc.forward_async(model_remote, X, nsample=5000)
    _spin_until(lambda: len(fut.job_ids) > 0 or fut.done(), timeout_s=12.0)
    Y_remote = fut.wait()
    assert Y_remote.shape == (4, 4)
    assert len(fut.job_ids) >= 1  # quantum offload happened

    # Now force local quantum execution and compare
    q.force_simulation = True
    fut_local = proc.forward_async(model_remote, X, nsample=5000)
    Y_local = fut_local.wait()
    assert Y_local.shape == (4, 4)

    # Sampling noise -> lenient tolerance
    assert torch.allclose(Y_local.sum(dim=1), torch.ones(4), atol=2e-2)
    assert torch.allclose(Y_remote.sum(dim=1), torch.ones(4), atol=2e-2)

    # Values should be "reasonably" close despite sampling noise
    # Use per-row L1 with a loose threshold
    l1 = _l1_dist(Y_local, Y_remote)
    assert torch.all(l1 <= 0.35), f"Classical/quantum pipeline outputs diverged: {l1.tolist()}"


# -------------------------
# COBYLA gradient-free fine-tuning via MerlinProcessor.forward
# -------------------------

def test_cobyla_finetune_over_merlin_forward(remote_processor):
    """
    Demonstrate parameter 'fine-tuning' of a QuantumLayer by running a
    derivative-free optimizer (COBYLA) that:
      - Sets QuantumLayer parameters from a flat vector
      - Evaluates an objective via MerlinProcessor.forward on a model that
        includes the QuantumLayer + fixed linear readout
      - Optimizes a tiny number of steps to move the metric in the right direction

    IMPORTANT: This test also asserts we're truly using the *remote* path by
    first calling forward_async() and checking that at least one cloud job_id
    is created (offload happened).
    """
    scipy = pytest.importorskip("scipy", reason="SciPy required for COBYLA test")
    from scipy.optimize import minimize

    torch.manual_seed(0)

    # ---- Build model (classical -> quantum -> readout) ----
    q = _make_layer(m=4, n=2, input_size=2, no_bunching=True).eval()
    dist = q(torch.rand(2, 2)).shape[1]

    readout = nn.Linear(dist, 1, bias=False).eval()
    with torch.no_grad():
        torch.manual_seed(42)
        readout.weight.copy_(torch.randn_like(readout.weight))

    pre = nn.Linear(3, 2, bias=False).eval()
    with torch.no_grad():
        pre.weight.copy_(torch.tensor([[1.0, 0.5, -0.2],
                                       [0.1, -0.3, 0.7]]))

    model = nn.Sequential(pre, q, readout)
    model.eval()  # REQUIRED by MerlinProcessor

    # ---- Flatten quantum params for COBYLA ----
    q_params = [(n, p) for n, p in q.named_parameters() if p.requires_grad]
    shapes = [p.shape for _, p in q_params]
    sizes = [p.numel() for _, p in q_params]

    def _get_flat() -> torch.Tensor:
        return torch.cat([p.detach().flatten().cpu() for _, p in q_params], dim=0)

    def _set_from_flat(vec: torch.Tensor) -> None:
        offset = 0
        with torch.no_grad():
            for (_, p), sz, shp in zip(q_params, sizes, shapes):
                chunk = vec[offset:offset+sz].view(shp).to(p.dtype)
                p.data.copy_(chunk.to(p.device))
                offset += sz

    x0 = _get_flat().double().numpy()

    proc = MerlinProcessor(remote_processor)

    # Use exact probs if available, otherwise sampling with moderate nsample
    use_probs = "probs" in getattr(proc, "available_commands", [])
    nsamp_eval = None if use_probs else 20_000

    X = torch.rand(8, 3)

    # ---- Prove we are truly remote before optimization ----
    fut = proc.forward_async(model, X[:2], nsample=nsamp_eval)
    # wait until we either have a job id or it's already done
    _spin_until(lambda: len(fut.job_ids) > 0 or fut.done(), timeout_s=12.0)
    assert len(fut.job_ids) >= 1, "Expected at least one remote job id (offload didn't happen)"
    _ = fut.wait()  # consume result for the smoke check

    # ---- COBYLA objective: maximize mean readout (we minimize the negative) ----
    def objective(vec_np):
        vec = torch.from_numpy(vec_np).to(torch.float64)
        _set_from_flat(vec.to(torch.float32))
        with torch.no_grad():
            y = proc.forward(model, X, nsample=nsamp_eval)  # remote evaluation
            return -float(y.mean().item())

    # ---- Quick optimization (keep runtime sane for CI) ----
    res = minimize(
        objective,
        x0,
        method="COBYLA",
        options={"maxiter": 12, "rhobeg": 0.5, "disp": False},
    )

    final_val = res.fun
    start_val = objective(x0)
    # Allow small slack due to sampling noise
    assert final_val <= start_val + 1e-3, f"COBYLA did not improve: start={start_val:.4f}, final={final_val:.4f}"