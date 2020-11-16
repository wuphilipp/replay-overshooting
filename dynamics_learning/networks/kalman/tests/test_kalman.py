import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn as nn
from torchdiffeq import odeint as odeint

from dynamics_learning.networks.kalman.ekf import EKFCell
from dynamics_learning.networks.models import SimpleODENetConfig
from dynamics_learning.utils.net_utils import batch_eye
from dynamics_learning.utils.tests.test_net_utils import wrap_zero_input


@given(batch_dim=st.integers(min_value=1, max_value=10),)
def test_EKF_basic(batch_dim: int):
    network = SimpleODENetConfig(
        input_dim=2,
        output_dim=2,
        hidden_layers=2,
        hidden_units=64,
        nonlinearity=nn.Tanh(),
    ).create()
    kf = EKFCell(network, None)
    x = torch.randn(batch_dim, 2)
    p0 = torch.randn(batch_dim, 2, 2)
    p0 = torch.bmm(p0, p0.transpose(-1, -2)) + batch_eye(2, batch_dim)

    vect = kf.gaussian_parameters_to_vector(x, p0)
    mu, var = kf.vector_to_gaussian_parameters(vect)

    # TODO uncomment
    # npt.assert_allclose(x.numpy(), mu.numpy())
    # npt.assert_allclose(p0.numpy(), var.numpy(), rtol=1e-6)


@given(batch_dim=st.integers(min_value=1, max_value=2),)
@settings(deadline=5000)
def test_EKFCell_dynamics_foward(batch_dim: int):
    network = SimpleODENetConfig(
        input_dim=2,
        output_dim=2,
        hidden_layers=2,
        hidden_units=2,
        nonlinearity=nn.Tanh(),
    ).create()
    kf_f = EKFCell(network, None)  # Model with cholesky

    batch_t = torch.linspace(0.0, 1.0, 4)
    x = torch.randn(batch_dim, 2)
    p0 = torch.randn(batch_dim, 2, 2)
    p0 = torch.bmm(p0, p0.transpose(-1, -2)) + batch_eye(2, batch_dim) * 10

    vect = kf_f.gaussian_parameters_to_vector(x, p0)
    xs_ps_1 = odeint(wrap_zero_input(kf_f), vect, batch_t)
    x1, var1 = kf_f.vector_to_gaussian_parameters(xs_ps_1)


# TODO: fix test - giving flaky resutls
# @given(batch_dim=st.integers(min_value=1, max_value=8),)
# @settings(deadline=5000)
# def test_EKFCell_dynamics_backwards(batch_dim: int):
#     network = SimpleODENetConfig(
#         input_dim=2,
#         output_dim=2,
#         hidden_layers=2,
#         hidden_units=64,
#         nonlinearity=nn.Tanh(),
#     ).create()
#     kf_f = EKFCell(network, None)  # Model with cholesky
#
#     batch_t = torch.linspace(0.0, 1.0, 5)
#
#     # TODO generate with hypothesis
#     x = torch.randn(batch_dim, 2)
#     p0 = torch.randn(batch_dim, 2, 2)
#     p0 = torch.bmm(p0, p0.transpose(-1, -2)) + batch_eye(2, batch_dim) * 10
#
#     vect = kf_f.gaussian_parameters_to_vector(x, p0)
#     xs_ps_1 = odeint(kf_f, vect, batch_t)
#     x1, var1 = kf_f.vector_to_gaussian_parameters(xs_ps_1)
#     fake_loss = torch.mean(x1 ** 2) + torch.mean(var1 ** 2)
#     fake_loss.backward()
