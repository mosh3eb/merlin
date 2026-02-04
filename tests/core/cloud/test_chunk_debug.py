"""
Diagnostic tests for chunking with ISession.

Investigates why chunk_concurrency=2 (waves) fails while
chunk_concurrency=4 (all at once) succeeds.

Run with:
    pytest --run-scaleway-tests tests/core/cloud/test_chunk_debug.py -v -s
"""

import time

import pytest
import torch
from math import comb

from merlin.algorithms import QuantumLayer
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.core.computation_space import ComputationSpace
from merlin.core.merlin_processor import MerlinProcessor


def _make_layer():
    b = CircuitBuilder(n_modes=6)
    b.add_rotations(trainable=True, name="theta")
    b.add_angle_encoding(modes=[0, 1], name="px")
    b.add_entangling_layer()
    return QuantumLayer(
        input_size=2,
        builder=b,
        n_photons=2,
        computation_space=ComputationSpace.UNBUNCHED,
    ).eval()


def _wait_future(fut, timeout=300.0):
    deadline = time.time() + timeout
    while not fut.done():
        if time.time() >= deadline:
            pytest.fail("Future timed out")
        time.sleep(0.5)
    return fut.value()


@pytest.mark.usefixtures("scaleway_session")
class TestChunkDebug:

    def test_sequential_chunks(self, scaleway_session):
        """chunk_concurrency=1: sequential, baseline."""
        proc = MerlinProcessor(
            session=scaleway_session,
            microbatch_size=8,
            timeout=300.0,
            max_shots_per_call=100,
            chunk_concurrency=1,
        )

        q = _make_layer()
        X = torch.rand(32, 2)  # 4 chunks of 8

        start = time.time()
        fut = proc.forward_async(q, X, nsample=100)
        y = _wait_future(fut)
        elapsed = time.time() - start

        assert y.shape == (32, comb(6, 2))
        print(f"\n  Sequential (concurrency=1): {elapsed:.1f}s")

    def test_parallel_all_at_once(self, scaleway_session):
        """chunk_concurrency=4: all 4 chunks launched together."""
        proc = MerlinProcessor(
            session=scaleway_session,
            microbatch_size=4,
            timeout=300.0,
            max_shots_per_call=100,
            chunk_concurrency=4,
        )

        q = _make_layer()
        X = torch.rand(16, 2)  # 4 chunks of 4

        start = time.time()
        fut = proc.forward_async(q, X, nsample=100)
        y = _wait_future(fut)
        elapsed = time.time() - start

        assert y.shape == (16, comb(6, 2))
        print(f"\n  All at once (concurrency=4): {elapsed:.1f}s")

    def test_parallel_waves(self, scaleway_session):
        """chunk_concurrency=2: 2 waves of 2 — this is the failing case."""
        proc = MerlinProcessor(
            session=scaleway_session,
            microbatch_size=8,
            timeout=300.0,
            max_shots_per_call=100,
            chunk_concurrency=2,
        )

        q = _make_layer()
        X = torch.rand(32, 2)  # 4 chunks of 8 -> 2 waves of 2

        start = time.time()
        fut = proc.forward_async(q, X, nsample=100)
        y = _wait_future(fut)
        elapsed = time.time() - start

        assert y.shape == (32, comb(6, 2))
        print(f"\n  Waves (concurrency=2): {elapsed:.1f}s")