# MIT License
#
# Copyright (c) 2025 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Passive bridge between qubit statevectors and Merlin photonic computation spaces.
"""

from __future__ import annotations

import itertools
from enum import Enum
from typing import Any, Iterable, Literal, Sequence

import perceval as pcvl
import torch
import torch.nn as nn

from merlin.torch_utils.dtypes import resolve_float_complex


class ComputationSpace(str, Enum):
    """Photonic computation spaces supported by the bridge."""

    FOCK = "fock"
    UNBUNCHED = "unbunched"
    DUAL_RAIL = "dual_rail"

    @classmethod
    def parse(cls, value: ComputationSpace | str) -> ComputationSpace:
        if isinstance(value, ComputationSpace):
            return value
        normalized = value.lower().replace("-", "_")
        try:
            return ComputationSpace(normalized)
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"Unknown computation space '{value}'. Choose from {choices}.") from exc


def to_fock_state(qubit_state: str, group_sizes: Sequence[int]) -> pcvl.BasicState:
    """
    Map a bitstring to a BasicState with one photon per qubit-group (one-hot over 2^k modes).
    No ancilla/postselected modes are added. The number of modes is m = Σ 2^group_size.
    """
    fock_state: list[int] = []
    bit_offset = 0
    for size in group_sizes:
        group_len = 2**size
        bits = qubit_state[bit_offset : bit_offset + size]
        idx = int(bits, 2)
        fock_state += [1 if i == idx else 0 for i in range(group_len)]
        bit_offset += size
    return pcvl.BasicState(fock_state)


def _to_occ_tuple(key: pcvl.BasicState | Sequence[int]) -> tuple[int, ...]:
    """Convert a BasicState or occupancy list to a tuple for dict keys."""
    if isinstance(key, pcvl.BasicState):
        return tuple(key)
    return tuple(key)


def _stars_and_bars(n_modes: int, n_photons: int) -> list[tuple[int, ...]]:
    """Enumerate Fock states (with bunching) using a lexicographic stars-and-bars scheme."""
    states: list[tuple[int, ...]] = []

    def recurse(mode: int, remaining: int, prefix: list[int]) -> None:
        if mode == n_modes - 1:
            prefix.append(remaining)
            states.append(tuple(prefix))
            prefix.pop()
            return
        for count in range(remaining + 1):
            prefix.append(count)
            recurse(mode + 1, remaining - count, prefix)
            prefix.pop()

    recurse(0, n_photons, [])
    return list(reversed(states))


def _unbunched_states(n_modes: int, n_photons: int) -> list[tuple[int, ...]]:
    """Enumerate unbunched states (at most one photon per mode)."""
    states: list[tuple[int, ...]] = []
    for combo in itertools.combinations(range(n_modes), n_photons):
        occ = [0] * n_modes
        for idx in combo:
            occ[idx] = 1
        states.append(tuple(occ))
    return states


class QuantumBridge(nn.Module):
    """
    Passive bridge between a qubit statevector (PyTorch tensor) and a Merlin QuantumLayer.

    The bridge applies a fixed transition matrix that maps computational-basis amplitudes
    into the selected photonic computation space (Fock, unbunched, or dual-rail).

    Args:
        n_photons: Number of logical photons (equals ``len(qubit_groups)``).
        n_modes: Total number of photonic modes that will be simulated downstream.
        qubit_groups: Logical grouping of qubits; ``[2, 1]`` means one photon is spread
            over ``2**2`` modes and another over ``2**1`` modes.
        wires_order: Endianness used to interpret computational basis strings.
        computation_space: Target photonic computation space. Accepts a
            :class:`ComputationSpace` enum or a string (``"fock"``, ``"unbunched"``,
            ``"dual_rail"``).
        normalize: Whether to L2-normalise input statevectors before applying the
            transition matrix.
        device: Optional device on which to place the output tensor.
        dtype: Real dtype that determines the corresponding complex dtype for amplitudes.
    """

    def __init__(
        self,
        n_photons: int,
        n_modes: int,
        *,
        qubit_groups: Sequence[int] | None = None,
        wires_order: Literal["little", "big"] = "little",
        computation_space: ComputationSpace | str = ComputationSpace.UNBUNCHED,
        normalize: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if wires_order not in ("little", "big"):
            raise ValueError("wires_order must be 'little' or 'big'.")

        self.computation_space = ComputationSpace.parse(computation_space)
        self._device = device
        self._dtype = dtype
        self.normalize = normalize

        if qubit_groups is None:
            if n_modes != 2 * n_photons:
                raise ValueError(
                    "If qubit_groups are omitted, n_modes must equal 2 * n_photons (dual-rail default)."
                )
            qubit_groups = [1] * n_photons
        if len(qubit_groups) != n_photons:
            raise ValueError(
                f"Length of qubit_groups ({len(qubit_groups)}) must match n_photons ({n_photons})."
            )

        self.group_sizes: tuple[int, ...] = tuple(int(g) for g in qubit_groups)
        self._n_photons = n_photons
        self._n_modes = n_modes
        self.wires_order = wires_order

        expected_modes = sum(2**g for g in self.group_sizes)
        if expected_modes != n_modes:
            raise ValueError(
                f"Provided n_modes={n_modes} incompatible with qubit_groups (expected {expected_modes})."
            )

        self.n_qubits = sum(self.group_sizes)
        self.expected_state_dim = 2**self.n_qubits
        self._norm_epsilon = 1e-12

        self._basis_occupancies = self._build_qloq_basis()
        self._output_basis = self._build_output_basis()
        self._index_map = {occ: idx for idx, occ in enumerate(self._output_basis)}

        missing = [occ for occ in self._basis_occupancies if occ not in self._index_map]
        if missing:
            raise ValueError(
                f"Selected computation space {self.computation_space.value} does not contain "
                f"the QLOQ occupancies produced by the qubit groups. "
                f"Example missing occupancy: {missing[0]}"
            )

        # Precompute sparse transition structure (rows, cols)
        row_indices: list[int] = []
        col_indices: list[int] = []
        for col, occ in enumerate(self._basis_occupancies):
            row = self._index_map[occ]
            row_indices.append(row)
            col_indices.append(col)

        if row_indices:
            indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
        else:
            indices = torch.empty((2, 0), dtype=torch.long)
        self._transition_indices = indices
        self._transition_shape = (len(self._output_basis), self.expected_state_dim)

    # ------------------------------------------------------------------
    # Basis construction
    # ------------------------------------------------------------------
    def _build_qloq_basis(self) -> tuple[tuple[int, ...], ...]:
        occupancies: list[tuple[int, ...]] = []
        for idx in range(self.expected_state_dim):
            bits = format(idx, f"0{self.n_qubits}b")
            if self.wires_order == "little":
                bits = bits[::-1]
            fock = to_fock_state(bits, self.group_sizes)
            occupancies.append(_to_occ_tuple(fock))
        return tuple(occupancies)

    def _build_output_basis(self) -> tuple[tuple[int, ...], ...]:
        if self.computation_space is ComputationSpace.DUAL_RAIL:
            if any(g != 1 for g in self.group_sizes):
                raise ValueError(
                    "Computation space 'dual_rail' requires qubit_groups to contain only ones."
                )
            return tuple(self._basis_occupancies)

        if self.computation_space is ComputationSpace.UNBUNCHED:
            states = _unbunched_states(self._n_modes, self._n_photons)
            return tuple(states)

        # FOCK
        states = _stars_and_bars(self._n_modes, self._n_photons)
        return tuple(states)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def basis_occupancies(self) -> tuple[tuple[int, ...], ...]:
        """QLOQ occupancies indexed like the computational basis."""
        return self._basis_occupancies

    @property
    def output_basis(self) -> tuple[tuple[int, ...], ...]:
        """Occupancies enumerating the selected computation space."""
        return self._output_basis

    @property
    def n_modes(self) -> int:
        return self._n_modes

    @property
    def n_photons(self) -> int:
        return self._n_photons

    # ------------------------------------------------------------------
    # Transition matrix
    # ------------------------------------------------------------------
    def transition_matrix(self, device: torch.device | None = None) -> torch.Tensor:
        """
        Sparse matrix mapping computational amplitudes (columns) to the computation-space ordering (rows).
        """
        target_device = (
            device if device is not None else self._device if self._device is not None else torch.device("cpu")
        )
        _, target_complex = resolve_float_complex(self._dtype)
        indices = self._transition_indices.to(device=target_device)
        values = torch.ones(
            indices.shape[1],
            dtype=target_complex,
            device=target_device,
        )
        transition = torch.sparse_coo_tensor(
            indices,
            values,
            size=self._transition_shape,
            dtype=target_complex,
            device=target_device,
        )
        return transition.coalesce()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, psi: torch.Tensor, *extra_args: Any) -> torch.Tensor:
        """
        Convert a qubit statevector ``psi`` into amplitudes ordered according to the computation space.
        """
        if extra_args:
            raise ValueError("QuantumBridge no longer forwards auxiliary arguments; received extra inputs.")
        if not isinstance(psi, torch.Tensor):
            raise TypeError("Statevector produced by the upstream module must be a torch.Tensor.")

        squeeze = False
        if psi.ndim == 1:
            psi = psi.unsqueeze(0)
            squeeze = True
        elif psi.ndim != 2:
            raise ValueError(
                f"QuantumBridge expects statevector shape (K,) or (B, K); received {psi.shape}."
            )

        _, target_complex = resolve_float_complex(self._dtype)
        target_device = self._device if self._device is not None else psi.device
        payload = psi.to(dtype=target_complex, device=target_device)

        if payload.shape[-1] != self.expected_state_dim:
            raise ValueError(
                f"Statevector dimension mismatch: expected {self.expected_state_dim}, "
                f"received {payload.shape[-1]}."
            )

        if self.normalize:
            norms = payload.norm(dim=1, keepdim=True)
            safe_norms = torch.where(
                norms > self._norm_epsilon,
                norms,
                torch.ones_like(norms),
            )
            payload = payload / safe_norms

        transition = self.transition_matrix(device=payload.device)
        transformed = torch.sparse.mm(transition, payload.transpose(0, 1)).transpose(0, 1)

        return transformed.squeeze(0) if squeeze else transformed


__all__ = ["ComputationSpace", "QuantumBridge", "to_fock_state"]
