import os

import perceval.providers.scaleway as scw
import perceval.providers.quandela as qc

from merlin.algorithms import QuantumLayer
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.core.merlin_processor import MerlinProcessor
from merlin.measurement.strategies import MeasurementStrategy

import torch
import torch.nn as nn


### Quandela session ###
TOKEN = os.environ["QC_TOKEN"]
PLATFORM_NAME = "sim:ascella"
with qc.Session(platform_name=PLATFORM_NAME, token=TOKEN) as session:

### Scaleway session ###
# PROJECT_ID = os.environ["SCW_PROJECT_ID"]
# TOKEN = os.environ["SCW_SECRET_KEY"]
# URL = os.environ["SCW_API_URL"]
# PLATFORM_NAME = "EMU-SAMPLING-L4"
# with scw.Session(platform_name=PLATFORM_NAME, project_id=PROJECT_ID, token=TOKEN, url=URL, max_idle_duration_s=300, max_duration_s=600) as session:

    proc = MerlinProcessor(
        session=session,
        microbatch_size=32,  # batch chunk size per cloud call (<=32)
        timeout=300.0,  # default wall-time per forward (seconds)
        max_shots_per_call=100,  # optional cap per cloud call (see below)
        chunk_concurrency=1,  # parallel chunk jobs within a quantum leaf
    )

    b = CircuitBuilder(n_modes=6)
    b.add_rotations(trainable=True, name="theta")
    b.add_angle_encoding(modes=[0, 1], name="px")
    b.add_entangling_layer()

    q = QuantumLayer(
        input_size=2,
        builder=b,
        n_photons=2,
        no_bunching=True,
        measurement_strategy=MeasurementStrategy.PROBABILITIES,  # raw probability vector
    ).eval()

    model = nn.Sequential(
        nn.Linear(3, 2, bias=False),
        q,
        nn.Linear(15, 4, bias=False),  # 15 = C(6,2) from the chosen circuit
        nn.Softmax(dim=-1),
    ).eval()

    X = torch.rand(8, 3)
    y = proc.forward(model, X, nsample=100)  # synchronous

print(y.shape)
