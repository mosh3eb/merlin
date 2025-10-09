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

import math

import torch

import merlin as ML
from merlin.sampling.strategies import GroupingPolicy, MeasurementStrategy


class TestMeasurementStrategy:
    def test_fock_distribution_equivalent_to_none(self):
        # FockDistribution is equivalent to OutputMappingStrategy.NONE
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL, n_modes=3, n_photons=1
        )
        output_size = math.comb(3, 1)
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=output_size,
            measurement_strategy=MeasurementStrategy.FOCKDISTRIBUTION,
        )
        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        x = torch.rand(2, 2)
        output = layer(x)
        assert output.shape == (2, 3)
        assert torch.all(output >= -1e6)

    def test_lexgrouping_equivalent(self):
        # LexGrouping is equivalent to FockGrouping + LexGrouping
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL, n_modes=4, n_photons=2
        )
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=4,
            measurement_strategy=MeasurementStrategy.FOCKGROUPING,
            grouping_policy=GroupingPolicy.LEXGROUPING,
        )
        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        x = torch.rand(3, 2)
        output = layer(x)
        assert output.shape == (3, 4)
        assert torch.all(torch.isfinite(output))

    def test_modgrouping_equivalent(self):
        # ModGrouping is equivalent to FockGrouping + ModGrouping
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL, n_modes=5, n_photons=2
        )
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=5,
            measurement_strategy=MeasurementStrategy.FOCKGROUPING,
            grouping_policy=GroupingPolicy.MODGROUPING,
        )
        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        x = torch.rand(4, 2)
        output = layer(x)
        assert output.shape == (4, 5)
        assert torch.all(torch.isfinite(output))

    def test_linear_equivalent(self):
        # OutputMappingStrategy.LINEAR is equivalent to FockDistribution + torch.nn.Linear
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL, n_modes=3, n_photons=1
        )
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            measurement_strategy=MeasurementStrategy.FOCKDISTRIBUTION,
        )
        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        linear = torch.nn.Linear(layer.output_size, 2)
        x = torch.rand(5, 2)
        output = linear(layer(x))
        assert output.shape == (5, 2)
        assert torch.all(torch.isfinite(output))

    def test_mode_expectation(self):
        # ModeExpectation
        n_modes = 3
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL, n_modes=n_modes, n_photons=1
        )
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=n_modes,
            measurement_strategy=MeasurementStrategy.MODEEXPECTATION,
        )
        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        x = torch.rand(2, 2)
        output = layer(x)
        assert output.shape == (2, 3)
        assert torch.all(torch.isfinite(output))

    def test_state_vector_equivalent(self):
        # StateVector is equivalent to return_amplitudes=True
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL, n_modes=3, n_photons=1
        )
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=3,
            measurement_strategy=MeasurementStrategy.STATEVECTOR,
        )
        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        x = torch.rand(2, 2)
        output = layer(x)
        assert output.shape == (2, 3)
        assert torch.all(torch.isfinite(output))
