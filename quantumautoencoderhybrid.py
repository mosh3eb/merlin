"""
Lightweight Hybrid Quantum Autoencoder using QuantumBridge
Uses the imported QuantumBridge class instead of manual implementation

This script:
- Encodes 4x4 patterns into a 4D latent (classical)
- Uses QuantumBridge to connect PennyLane circuit to Merlin photonic layer
- Decodes back to 16D and trains with MSE

Requirements:
- torch, numpy, matplotlib, pennylane, perceval, merlin
- QuantumBridge module (from the provided bridge implementation)
"""

import os
import random

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")  # safe for headless environments
import matplotlib.pyplot as plt
import pennylane as qml
import perceval as pcvl

from bridge import QuantumBridge  # Import the bridge class

# Merlin layer and bridge
from merlin import OutputMappingStrategy, QuantumLayer


# ----------------------------
# Seeding for reproducibility
# ----------------------------
def set_global_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_global_seed(0)


# ----------------------------
# PennyLane Module Wrapper
# ----------------------------
class PennyLaneModule(nn.Module):
    """Wrapper to make PennyLane circuit compatible with QuantumBridge"""

    def __init__(self, n_qubits: int, dtype=torch.float32, device=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.dtype = dtype
        self.device = device

        # Create PennyLane device and parameters
        self.pennylane_device = qml.device("default.qubit", wires=n_qubits)
        self.qubit_params = nn.Parameter(
            torch.randn(n_qubits, 2, dtype=dtype, device=device) * 0.1
        )

        # Build the QNode
        self.qnode = self._create_circuit()

    def _create_circuit(self):
        @qml.qnode(self.pennylane_device, interface="torch", diff_method="backprop")
        def circuit(inputs, params):
            # 3 lightweight layers
            for _ in range(3):
                n = min(inputs.shape[0], self.n_qubits)
                for i in range(n):
                    qml.RY(inputs[i] * np.pi, wires=i)
                for i in range(self.n_qubits):
                    qml.RZ(params[i, 0], wires=i)
                    qml.RX(params[i, 1], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.state()  # complex vector of length 2^n_qubits

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through quantum circuit
        Args:
            x: Input tensor of shape (batch_size, features) or (features,)
        Returns:
            Complex statevector tensor
        """
        if x.ndim == 1:
            # Single sample
            if x.shape[0] < self.n_qubits:
                x = F.pad(x, (0, self.n_qubits - x.shape[0]))
            else:
                x = x[:self.n_qubits]
            return self.qnode(x, self.qubit_params)

        # Batch processing
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            inp = x[i]
            if inp.shape[0] < self.n_qubits:
                inp = F.pad(inp, (0, self.n_qubits - inp.shape[0]))
            else:
                inp = inp[:self.n_qubits]
            state = self.qnode(inp, self.qubit_params)
            outputs.append(state)

        return torch.stack(outputs)


# ----------------------------
# Create photonic circuit helper
# ----------------------------
def create_photonic_circuit(n_modes: int):
    """
    Create a photonic circuit with two generic interferometers

    Args:
        n_modes: Number of optical modes
    Returns:
        Perceval circuit
    """

    def gi(side: str):
        return pcvl.GenericInterferometer(
            n_modes,
            lambda idx: pcvl.BS(theta=pcvl.P(f"theta_{side}{idx}")),
            shape=pcvl.InterferometerShape.RECTANGLE,
            depth=2 * n_modes,
        )

    circuit = pcvl.Circuit(n_modes)
    circuit.add(0, gi("l"), merge=True)
    circuit.add(0, gi("r"), merge=True)
    return circuit


# -------------------------
# The hybrid autoencoder
# -------------------------
class LightweightQuantumAutoencoder(nn.Module):
    """Quantum autoencoder using QuantumBridge"""

    def __init__(self, input_dim=16, latent_dim=4):
        super().__init__()

        # Very simple classical encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Sigmoid()
        )

        # Setup for quantum bridge
        qubit_groups = [2, 2]  # 4 qubits in 2 groups
        n_qubits = sum(qubit_groups)
        n_photons = len(qubit_groups)
        n_photonic_modes = sum(2 ** g for g in qubit_groups)  # 4 + 4 = 8 modes

        # Create PennyLane module
        self.pl_module = PennyLaneModule(n_qubits=n_qubits)

        # Create Merlin layer (photonic circuit)
        photonic_circuit = create_photonic_circuit(n_photonic_modes)

        self.merlin_layer = QuantumLayer(
            input_size=0,  # No classical input encoding
            output_size=latent_dim,
            circuit=photonic_circuit,
            n_photons=n_photons,
            trainable_parameters=["theta_"],
            input_parameters=[],
            output_mapping_strategy=OutputMappingStrategy.LINEAR,
            no_bunching=True,
        )

        # Create the quantum bridge
        self.quantum_bridge = QuantumBridge(
            qubit_groups=qubit_groups,
            merlin_layer=self.merlin_layer,
            pl_module=self.pl_module,
            wires_order="little",
            normalize=True,
        )

        # Simple decoder
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        # Encode to latent
        latent = self.encoder(x)

        # Process through quantum bridge
        quantum = self.quantum_bridge(latent)

        # Decode back
        reconstructed = torch.sigmoid(self.decoder(quantum))

        return reconstructed, latent, quantum


# -------------------------
# Data, training, plotting (unchanged)
# -------------------------
def generate_simple_patterns(n_samples=100, size=4):
    """Generate very simple 4x4 patterns"""
    patterns = []
    labels = []

    for _ in range(n_samples):
        pattern_type = np.random.choice(["horizontal", "vertical", "diagonal", "checkerboard"])
        img = np.zeros((size, size))

        if pattern_type == "horizontal":
            img[np.random.randint(0, size), :] = 1
            labels.append(0)
        elif pattern_type == "vertical":
            img[:, np.random.randint(0, size)] = 1
            labels.append(1)
        elif pattern_type == "diagonal":
            np.fill_diagonal(img, 1)
            labels.append(2)
        elif pattern_type == "checkerboard":
            img[::2, ::2] = 1
            img[1::2, 1::2] = 1
            labels.append(3)

        # Add small noise
        img += np.random.randn(size, size) * 0.05
        img = np.clip(img, 0, 1)

        patterns.append(img.flatten())

    return torch.tensor(patterns, dtype=torch.float32), torch.tensor(labels)


def visualize_results(model, test_data, test_labels, n_samples=8):
    """Visualize reconstruction and quantum states"""
    model.eval()

    fig = plt.figure(figsize=(14, 8))

    with torch.no_grad():
        reconstructed, latent, quantum = model(test_data[:n_samples])

        # Plot originals and reconstructions
        for i in range(n_samples):
            # Original
            ax = plt.subplot(4, n_samples, i + 1)
            orig = test_data[i].reshape(4, 4).numpy()
            ax.imshow(orig, cmap="coolwarm", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel("Original", fontsize=8)

            # Reconstructed
            ax = plt.subplot(4, n_samples, n_samples + i + 1)
            recon = reconstructed[i].reshape(4, 4).numpy()
            ax.imshow(recon, cmap="coolwarm", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel("Reconstructed", fontsize=8)

            # Latent representation
            ax = plt.subplot(4, n_samples, 2 * n_samples + i + 1)
            ax.bar(range(len(latent[i])), latent[i].numpy())
            ax.set_ylim([0, 1])
            ax.axis("off")
            if i == 0:
                ax.set_ylabel("Latent", fontsize=8)

            # Quantum output
            ax = plt.subplot(4, n_samples, 3 * n_samples + i + 1)
            ax.bar(range(len(quantum[i])), quantum[i].numpy())
            ax.set_ylim([0, 1])
            ax.axis("off")
            if i == 0:
                ax.set_ylabel("Quantum", fontsize=8)

    plt.suptitle("Quantum Autoencoder: Original → Latent → Quantum → Reconstructed")
    plt.tight_layout()
    return fig


def train_autoencoder(model, train_data, val_data, epochs=150, lr=0.05, clip_grad=1.0):
    """Training loop"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        reconstructed, _, _ = model(train_data)
        train_loss = criterion(reconstructed, train_data)

        train_loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        train_losses.append(train_loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_reconstructed, _, _ = model(val_data)
            val_loss = criterion(val_reconstructed, val_data).item()

        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss.item():.6f}, Val Loss = {val_loss:.6f}")

    return train_losses, val_losses


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    print("Quantum Autoencoder using QuantumBridge\n")
    print("Seamless integration of PennyLane and Merlin via imported bridge\n")

    # Generate data
    train_data, train_labels = generate_simple_patterns(200, size=4)
    val_data, val_labels = generate_simple_patterns(50, size=4)

    # Create model
    model = LightweightQuantumAutoencoder(input_dim=16, latent_dim=4)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    quantum_params = sum(p.numel() for p in model.quantum_bridge.parameters())

    print("Model configuration:")
    print("  Qubits: 4 (groups: [2, 2])")
    print("  Photons: 2")
    print("  Photonic modes: 8")
    print(f"  Parameters: {total_params} (quantum bridge: {quantum_params})\n")

    # Train
    print("Training...")
    train_losses, val_losses = train_autoencoder(model, train_data, val_data, epochs=150, lr=0.05)

    # Visualize
    os.makedirs("outputs", exist_ok=True)
    print("\nGenerating visualizations...")
    fig = visualize_results(model, val_data, val_labels)
    figpath1 = os.path.join("outputs", "quantum_autoencoder_results.png")
    plt.savefig(figpath1, dpi=110)
    plt.close(fig)

    # Plot training history
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train", alpha=0.9)
    plt.plot(val_losses, label="Validation", alpha=0.9)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    figpath2 = os.path.join("outputs", "training_history.png")
    plt.savefig(figpath2, dpi=110)
    plt.close()

    print("\nComplete! Saved visualizations:")
    print(f"  - {figpath1}")
    print(f"  - {figpath2}")
