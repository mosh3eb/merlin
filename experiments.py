import numpy as np
import pennylane as qml

dev = qml.device("default.qubit", wires=2)


def RBS(theta, wire_1, wire_2):
    """
    Applies the RBS onto the qml (Pennylane) circuit.

    This is the main gate used for vector loading onto a quantum circuit.

    Args:
        theta: Parameter of the RBS
        wire_1: First wire on which the RBS is applied
        wire_2: Second wire on which the RBS is applied

    Returns:

    """
    qml.H(wire_1)
    qml.H(wire_2)
    qml.CZ(wires=[wire_1, wire_2])
    # qml.RY(theta, wires=wire_1)
    # qml.RY(-theta, wires=wire_2)
    qml.RY(theta / 2, wires=wire_1)
    qml.RY(-theta / 2, wires=wire_2)
    qml.CZ(wires=[wire_1, wire_2])
    qml.H(wire_1)
    qml.H(wire_2)


@qml.qnode(dev)
def exp_RBS(theta):
    qml.PauliX(0)
    RBS(theta, 0, 1)
    return qml.state()


states = exp_RBS(np.pi / 4)

# enumerate over amplitudes
n_wires = 2
basis_states = [format(i, f"0{n_wires}b") for i in range(2**n_wires)]

for b, amp in zip(basis_states, states, strict=False):
    print(f"|{b}>: {amp}")
