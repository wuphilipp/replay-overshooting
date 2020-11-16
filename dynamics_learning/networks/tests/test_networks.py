import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from torch import nn as nn

from dynamics_learning.networks.models import SimpleODENetConfig


def test_simple_net_construction():
    network_config = SimpleODENetConfig(
        input_dim=2,
        output_dim=2,
        hidden_layers=2,
        hidden_units=64,
        nonlinearity=nn.Tanh(),
    )
    network_config.create()

    network_config = SimpleODENetConfig(
        input_dim=4,
        output_dim=4,
        hidden_layers=4,
        hidden_units=16,
        nonlinearity=nn.ReLU(),
    )
    network_config.create()


@given(
    network_input=hnp.arrays(shape=(5, 2), dtype=np.float).filter(
        lambda x: np.linalg.norm(x) < np.inf
    ),
    output_dim=st.integers(min_value=1, max_value=10),
)
def test_simple_net_compute(network_input: np.array, output_dim: int):
    B, _ = network_input.shape
    network = SimpleODENetConfig(
        input_dim=2,
        output_dim=output_dim,
        hidden_layers=2,
        hidden_units=64,
        nonlinearity=nn.Tanh(),
    ).create()
    network_input = torch.tensor(network_input, dtype=torch.float)
    t = network_input[:, 0:1]
    x = network_input
    network_output = network(t, x)
    assert B, output_dim == network_output.shape
