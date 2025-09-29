Quantum Bridge (PennyLane ↔ Merlin)
===================================

Overview
--------
The Quantum Bridge lets you plug a PennyLane-style state preparation into a Merlin `QuantumLayer` by mapping qubit basis states into photonic Fock states using a one-photon-per-group encoding. This enables hybrid models where the differentiable statevector comes from a qubit simulator, and the photonic circuit and measurement are handled by Merlin.

Key ideas:
- You provide a preconfigured `QuantumLayer` (we do not build the circuit in the bridge).
- You provide either a `pl_module` (PyTorch nn.Module) or `pl_state_fn` (callable) that returns a complex statevector of size 2^n.
- You partition the n qubits into groups via `qubit_groups` (e.g., [2, 2] for 4 qubits), each group mapping to one photon spread over 2^group_size modes.
- The bridge scatters amplitudes into Merlin's `mapped_keys` ordering and uses the superposition path for differentiability (no sampling).

Minimal example
---------------
.. code-block:: python

    import torch
    import perceval as pcvl
    from merlin import QuantumLayer, OutputMappingStrategy
    from merlin.bridge.QuantumBridge import QuantumBridge

    # Build a simple identity photonic circuit with m = sum(2**g) modes
    qubit_groups = [1, 1]  # two groups of one qubit each → 2 photons, 4 modes
    m = sum(2**g for g in qubit_groups)

    circuit = pcvl.Circuit(m)
    layer = QuantumLayer(
        input_size=0,
        circuit=circuit,
        n_photons=len(qubit_groups),
        output_mapping_strategy=OutputMappingStrategy.NONE,
        no_bunching=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # PennyLane-like state provider: returns a complex statevector of length 2^n
    def pl_state_fn(x: torch.Tensor) -> torch.Tensor:
        psi = torch.zeros(4, dtype=torch.complex64)
        psi[1] = 1 + 0j  # |01>
        return psi

    bridge = QuantumBridge(
        qubit_groups=qubit_groups,
        merlin_layer=layer,
        pl_state_fn=pl_state_fn,
        wires_order="little",  # or "big"
        normalize=True,         # L2-normalize the input state
    )

    x = torch.zeros(1, 1)  # dummy input if your state fn ignores it
    y = bridge(x)          # probability distribution over photonic outcomes

Devices and dtypes
------------------
- The output device follows the `QuantumLayer` device. Ensure the layer and bridge use the same device (CPU/CUDA).
- The bridge converts input states to complex64 for float32 and to complex128 for float64. The distribution dtype is the `QuantumLayer` real dtype (float32/float64).

Constraints
-----------
- No ancilla or postselected modes; total modes m must equal sum(2**group_size).
- Number of photons equals the number of groups.
- `QuantumLayer` must be provided; the bridge does not create circuits.

API
---
.. autofunction:: merlin.bridge.QuantumBridge.to_fock_state

.. automodule:: merlin.bridge.QuantumBridge
   :members: QuantumBridge
   :undoc-members:
   :show-inheritance:
