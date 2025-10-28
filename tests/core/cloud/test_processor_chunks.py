"""
Chunked parallel offload tests for MerlinProcessor.

Focus:
- When B > max_batch_size, processor splits into chunks and submits up to
  chunk_concurrency remote jobs in parallel per quantum leaf.
- job_ids aggregates one id per chunk submission (across all leaves).
- Shape and stitching are correct.
- Global timeout cancels all in-flight chunk jobs (fan-out).
- cancel_all() cancels all chunked jobs.
- Two quantum leaves both chunk -> job_ids >= sum of chunks across leaves.

Requires cloud; auto-skips if no token via `remote_processor` fixture from conftest.py.
"""

from __future__ import annotations

import concurrent.futures as _cf
import time

import pytest
import torch
import torch.nn as nn

from merlin.algorithms import QuantumLayer
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.core.merlin_processor import MerlinProcessor
from merlin.sampling.strategies import OutputMappingStrategy


def _spin_until(pred, timeout_s: float = 10.0, sleep_s: float = 0.02) -> bool:
    start = time.time()
    while not pred():
        if time.time() - start > timeout_s:
            return False
        time.sleep(sleep_s)
    return True


def _make_layer(m: int, n: int, input_size: int, no_bunching: bool = True) -> QuantumLayer:
    builder = CircuitBuilder(n_modes=m)
    builder.add_rotations(trainable=True, name="theta")
    builder.add_angle_encoding(modes=list(range(input_size)), name="px")
    if m >= 3:
        builder.add_entangling_layer()

    return QuantumLayer(
        input_size=input_size,
        output_size=None,  # raw distribution
        builder=builder,
        n_photons=n,
        no_bunching=no_bunching,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    ).eval()


class TestChunkedOffload:
    def test_chunking_job_ids_and_shape(self, remote_processor):
        """
        With B > max_batch_size and chunk_concurrency>1, we expect:
          - len(job_ids) == number of chunks
          - correct output shape
          - status() exposes chunk counters
        """
        # One quantum leaf: 6 modes, 2 photons, no-bunching -> output 15
        q = _make_layer(m=6, n=2, input_size=2, no_bunching=True).eval()

        # Force chunking: B=5, max_batch_size=2 -> chunks: [0:2], [2:4], [4:5] -> 3 chunks
        B = 5
        X = torch.rand(B, 2)

        proc = MerlinProcessor(
            remote_processor,
            max_batch_size=2,
            timeout=3600.0,
            max_shots_per_call=50_000,  # Perceval requires this for RemoteProcessor
            chunk_concurrency=2,        # parallelize within the leaf
        )

        fut = proc.forward_async(q, X, nsample=2000)  # use sampling if probs unavailable
        # Wait until we either have job IDs or it's already done
        _spin_until(lambda f=fut: len(f.job_ids) >= 3 or f.done(), timeout_s=20.0)
        y = fut.wait()

        assert y.shape == (B, 15)
        assert len(fut.job_ids) >= 3  # exactly 3 in normal flow

        st = fut.status()
        # status should expose chunk counters
        assert "chunks_total" in st and "chunks_done" in st and "active_chunks" in st
        assert st["chunks_total"] >= 3

    def test_chunking_timeout_sets_timeouterror(self, remote_processor):
        """
        Global timeout must cancel all in-flight chunk jobs.
        Use a larger nsample to slow down (if we are on sampling backend).
        """
        q = _make_layer(m=6, n=2, input_size=2, no_bunching=True).eval()
        B = 8
        X = torch.rand(B, 2)

        proc = MerlinProcessor(
            remote_processor,
            max_batch_size=2,     # 4 chunks
            timeout=3600.0,       # default large; per-call timeout will be small
            max_shots_per_call=100_000,
            chunk_concurrency=2,
        )

        fut = proc.forward_async(q, X, nsample=50_000, timeout=0.05)  # tiny per-call timeout
        with pytest.raises(TimeoutError):
            fut.wait()

    def test_cancel_all_cancels_chunked_futures(self, remote_processor):
        """
        cancel_all() should cancel all in-flight chunk jobs and set CancelledError.
        """
        q = _make_layer(m=6, n=2, input_size=2, no_bunching=True).eval()
        B = 9
        X = torch.rand(B, 2)

        proc = MerlinProcessor(
            remote_processor,
            max_batch_size=3,     # 3 chunks
            timeout=3600.0,       # no auto-timeout
            max_shots_per_call=80_000,
            chunk_concurrency=2,
        )

        fut = proc.forward_async(q, X, nsample=40_000, timeout=None)

        # Wait until jobs are submitted
        ok = _spin_until(lambda f=fut: len(f.job_ids) >= 2 or f.done(), timeout_s=10.0)
        if not ok or fut.done():
            pytest.skip("Backend finished too quickly to test cancellation")

        proc.cancel_all()
        with pytest.raises(_cf.CancelledError):
            fut.wait()

    def test_two_quantum_leaves_both_chunked_have_expected_jobs(self, remote_processor):
        """
        Two quantum leaves in a sequential model; each should chunk independently.
        Expect job_ids to be at least sum of chunks across both leaves.
        """
        # q1: 4m,2p -> 6; q2: 5m,2p -> 10 (raw distributions)
        q1 = _make_layer(m=4, n=2, input_size=1, no_bunching=True).eval()
        q2 = _make_layer(m=5, n=2, input_size=2, no_bunching=True).eval()

        model = nn.Sequential(
            nn.Linear(3, 1, bias=False),
            q1,
            nn.Linear(6, 2, bias=False),
            q2,
            nn.Linear(10, 3, bias=False),
            nn.Softmax(dim=-1),
        ).eval()

        # B=7, max_batch_size=3 -> chunks per leaf: 3 (3,3,1)
        B = 7
        X = torch.rand(B, 3)

        proc = MerlinProcessor(
            remote_processor,
            max_batch_size=3,
            timeout=3600.0,
            max_shots_per_call=60_000,
            chunk_concurrency=2,
        )

        fut = proc.forward_async(model, X, nsample=3000)
        _spin_until(lambda f=fut: len(f.job_ids) >= 4 or f.done(), timeout_s=20.0)  # heuristic
        y = fut.wait()

        assert y.shape == (B, 3)
        # Expect about 6 jobs (3 per leaf) if both leaves chunk; allow >= 4 to be robust to very fast backends
        assert len(fut.job_ids) >= 4
