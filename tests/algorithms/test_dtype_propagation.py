#!/usr/bin/env python3
"""
Test dtype propagation fix for ModeExpectations mask.

Run from project root:
    python tests/test_dtype_propagation.py
"""

import torch
import pytest

from merlin.core import ComputationSpace
from merlin.measurement import ModeExpectations, OutputMapper
from merlin.measurement.strategies import MeasurementStrategy


class TestModeExpectationsDtype:
    """Test that ModeExpectations respects the dtype parameter."""

    KEYS = [(0, 1), (1, 0), (1, 1), (0, 0)]

    def test_default_dtype_is_float32(self):
        """Without dtype argument, mask should default to float32."""
        mapper = ModeExpectations(ComputationSpace.UNBUNCHED, self.KEYS)
        assert mapper.mask.dtype == torch.float32

    def test_explicit_float32(self):
        """Explicit float32 should work."""
        mapper = ModeExpectations(
            ComputationSpace.UNBUNCHED, self.KEYS, dtype=torch.float32
        )
        assert mapper.mask.dtype == torch.float32

    def test_explicit_float64(self):
        """Explicit float64 should create float64 mask."""
        mapper = ModeExpectations(
            ComputationSpace.UNBUNCHED, self.KEYS, dtype=torch.float64
        )
        assert mapper.mask.dtype == torch.float64

    def test_forward_preserves_float64(self):
        """Forward pass with float64 input should not raise dtype mismatch."""
        mapper = ModeExpectations(
            ComputationSpace.UNBUNCHED, self.KEYS, dtype=torch.float64
        )
        x = torch.rand(2, 4, dtype=torch.float64)
        result = mapper(x)
        assert result.dtype == torch.float64

    def test_forward_preserves_float32(self):
        """Forward pass with float32 input should work."""
        mapper = ModeExpectations(
            ComputationSpace.UNBUNCHED, self.KEYS, dtype=torch.float32
        )
        x = torch.rand(2, 4, dtype=torch.float32)
        result = mapper(x)
        assert result.dtype == torch.float32

    def test_fock_space_float64(self):
        """FOCK computation space should also respect dtype."""
        mapper = ModeExpectations(
            ComputationSpace.FOCK, self.KEYS, dtype=torch.float64
        )
        assert mapper.mask.dtype == torch.float64

    def test_dual_rail_float64(self):
        """DUAL_RAIL computation space should also respect dtype."""
        mapper = ModeExpectations(
            ComputationSpace.DUAL_RAIL, self.KEYS, dtype=torch.float64
        )
        assert mapper.mask.dtype == torch.float64


class TestOutputMapperDtype:
    """Test that OutputMapper.create_mapping passes dtype correctly."""

    KEYS = [(0, 1), (1, 0), (1, 1), (0, 0)]

    def test_create_mapping_without_dtype(self):
        """Factory without dtype should create float32 mask."""
        mapper = OutputMapper.create_mapping(
            MeasurementStrategy.MODE_EXPECTATIONS,
            ComputationSpace.UNBUNCHED,
            self.KEYS,
        )
        assert mapper.mask.dtype == torch.float32

    def test_create_mapping_with_float64(self):
        """Factory with dtype=float64 should create float64 mask."""
        mapper = OutputMapper.create_mapping(
            MeasurementStrategy.MODE_EXPECTATIONS,
            ComputationSpace.UNBUNCHED,
            self.KEYS,
            dtype=torch.float64,
        )
        assert mapper.mask.dtype == torch.float64

    def test_create_mapping_with_float32(self):
        """Factory with explicit dtype=float32 should create float32 mask."""
        mapper = OutputMapper.create_mapping(
            MeasurementStrategy.MODE_EXPECTATIONS,
            ComputationSpace.UNBUNCHED,
            self.KEYS,
            dtype=torch.float32,
        )
        assert mapper.mask.dtype == torch.float32

    def test_probabilities_ignores_dtype(self):
        """Probabilities strategy should work regardless of dtype arg."""
        mapper = OutputMapper.create_mapping(
            MeasurementStrategy.PROBABILITIES,
            dtype=torch.float64,
        )
        # Just verify it doesn't crash; Probabilities has no mask
        assert mapper is not None

    def test_amplitudes_ignores_dtype(self):
        """Amplitudes strategy should work regardless of dtype arg."""
        mapper = OutputMapper.create_mapping(
            MeasurementStrategy.AMPLITUDES,
            dtype=torch.float64,
        )
        assert mapper is not None


class TestEndToEndDtypeMismatch:
    """Test the actual failure case that was reported."""

    KEYS = [(0, 1), (1, 0), (1, 1), (0, 0)]

    def test_float64_matmul_succeeds(self):
        """The original bug: float64 input @ float32 mask would crash."""
        mapper = ModeExpectations(
            ComputationSpace.UNBUNCHED, self.KEYS, dtype=torch.float64
        )

        # Simulate what QuantumLayer does: creates float64 probability distribution
        prob_distribution = torch.rand(4, 4, dtype=torch.float64)

        # This would previously raise:
        # RuntimeError: expected m1 and m2 to have the same dtype, but got: double != float
        result = mapper.marginalize_per_mode(prob_distribution)

        assert result.dtype == torch.float64
        assert result.shape == (4, 2)  # (batch, num_modes)

    def test_mismatched_dtype_raises(self):
        """Verify mismatched dtypes still raise (sanity check)."""
        mapper = ModeExpectations(
            ComputationSpace.UNBUNCHED, self.KEYS, dtype=torch.float32
        )
        prob_distribution = torch.rand(4, 4, dtype=torch.float64)

        with pytest.raises(RuntimeError, match="expected .* same dtype"):
            mapper.marginalize_per_mode(prob_distribution)


def run_quick_validation():
    """Run a quick validation without pytest."""
    print("Running dtype propagation validation...\n")

    keys = [(0, 1), (1, 0), (1, 1), (0, 0)]
    errors = []

    # Test 1: Default dtype
    mapper = ModeExpectations(ComputationSpace.UNBUNCHED, keys)
    if mapper.mask.dtype != torch.float32:
        errors.append(f"FAIL: Default dtype is {mapper.mask.dtype}, expected float32")
    else:
        print("✓ Default dtype is float32")

    # Test 2: Explicit float64
    mapper64 = ModeExpectations(ComputationSpace.UNBUNCHED, keys, dtype=torch.float64)
    if mapper64.mask.dtype != torch.float64:
        errors.append(f"FAIL: Explicit float64 gave {mapper64.mask.dtype}")
    else:
        print("✓ Explicit float64 creates float64 mask")

    # Test 3: Forward pass with float64
    x = torch.rand(2, 4, dtype=torch.float64)
    try:
        result = mapper64(x)
        if result.dtype != torch.float64:
            errors.append(f"FAIL: Output dtype is {result.dtype}, expected float64")
        else:
            print("✓ Forward pass with float64 succeeds")
    except RuntimeError as e:
        errors.append(f"FAIL: Forward pass raised: {e}")

    # Test 4: OutputMapper factory
    factory_mapper = OutputMapper.create_mapping(
        MeasurementStrategy.MODE_EXPECTATIONS,
        ComputationSpace.UNBUNCHED,
        keys,
        dtype=torch.float64,
    )
    if factory_mapper.mask.dtype != torch.float64:
        errors.append(f"FAIL: Factory with float64 gave {factory_mapper.mask.dtype}")
    else:
        print("✓ OutputMapper.create_mapping passes dtype correctly")

    # Test 5: The original bug scenario
    prob = torch.rand(4, 4, dtype=torch.float64)
    try:
        result = factory_mapper.marginalize_per_mode(prob)
        print("✓ float64 matmul succeeds (original bug is fixed)")
    except RuntimeError as e:
        errors.append(f"FAIL: Original bug still present: {e}")

    print()
    if errors:
        print("=" * 60)
        print("FAILURES:")
        for err in errors:
            print(f"  {err}")
        print("=" * 60)
        return 1
    else:
        print("=" * 60)
        print("All validations passed! The dtype fix is working correctly.")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    import sys

    # Try pytest first, fall back to manual validation
    try:
        sys.exit(pytest.main([__file__, "-v"]))
    except Exception:
        sys.exit(run_quick_validation())