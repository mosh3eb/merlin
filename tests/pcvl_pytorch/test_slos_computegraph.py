# MIT License
#
# Copyright (c) 2025 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
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

import torch
from perceval.components import BS

from merlin.pcvl_pytorch.slos_torchscript import build_slos_distribution_computegraph


def test_slos_compute_probs_from_amplitudes_normalizes():
    unitary = torch.tensor(BS().compute_unitary()).unsqueeze(0)
    unitary = unitary.to(torch.complex64)

    graph = build_slos_distribution_computegraph(m=2, n_photons=2, dtype=torch.float)

    _, amplitudes = graph.compute(unitary, [1, 1])
    _, probabilities = graph.compute_probs_from_amplitudes(amplitudes)

    assert torch.allclose(
        probabilities.sum(), torch.tensor(1.0, dtype=probabilities.dtype), atol=1e-6
    )
