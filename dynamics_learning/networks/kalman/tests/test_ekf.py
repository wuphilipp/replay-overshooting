import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torchdiffeq import odeint

from dynamics_learning.networks.kalman.ekf import EKFCell
from dynamics_learning.networks.kalman.tests.utils import (
    LinearSystem,
    assert_tensor_symmetric_psd,
)
from dynamics_learning.utils.tests.test_net_utils import wrap_zero_input

MAX_EXAMPLES = 10


@given(data=st.data())
@pytest.mark.parametrize("is_continuous", [True, False])
@settings(deadline=5000, print_blob=True, max_examples=MAX_EXAMPLES)
def test_EKFCell_forward(data: st.DataObject, is_continuous: bool):
    batch_dim = data.draw(st.integers(min_value=1, max_value=16))
    torch.manual_seed(data.draw(st.integers(min_value=0, max_value=10)))

    network = LinearSystem(2, 2)
    ekf = EKFCell(network, None, is_continuous=is_continuous)
    ekf_z0 = ekf.get_initial_hidden_state(batch_dim)
    if is_continuous:
        ekf_forward = odeint(wrap_zero_input(ekf), ekf_z0, ekf_z0.new_tensor([0, 0.1]))[
            1, :, :
        ]
    else:
        ekf_forward = ekf.forward(
            ekf_z0.new_tensor([0]), ekf_z0, torch.zeros(batch_dim, 0)
        )
    mean, cov = ekf.vector_to_gaussian_parameters(ekf_forward)
    assert_tensor_symmetric_psd(cov)


# @given(data=st.data())
# @pytest.mark.parametrize("is_continuous", [True, False])
# @settings(deadline=5000, print_blob=True, max_examples=10)
# def test_EKFCell_measurement_update_only(data: st.DataObject, is_continuous: bool):
#     batch_dim = data.draw(st.integers(min_value=1, max_value=16))
#     torch.manual_seed(data.draw(st.integers(min_value=0, max_value=10)))
#
#     network = LinearSystem(2, 2)
#     obs = nn.Linear(2, 2)
#     ekf = EKFCell(
#         network,
#         obs,
#         is_continuous=is_continuous,
#         initial_variance=np.eye(2),
#         process_noise=np.eye(2),
#         measurement_noise=np.eye(2),
#     )
#
#     z0 = data.draw(tensor_strategy((batch_dim, 2)), "z0")
#     ekf_z0 = ekf.get_initial_hidden_state(batch_dim, z0)
#
#     y = data.draw(tensor_strategy((batch_dim, 2)), "y")
#     ekf_measurement = ekf.measurement_update(ekf_z0.new_tensor([0]), y, ekf_z0)
#
#     mean, cov = ekf.vector_to_gaussian_parameters(ekf_measurement)
#     assert_tensor_symmetric_psd(cov)


########################################################################################


# @given(data=st.data())
# @pytest.mark.parametrize("is_continuous", [True, False])
# @settings(deadline=5000, print_blob=True, max_examples=10)
# def test_EKFEstimator_forward(data: st.DataObject, is_continuous: bool):
#     batch_dim = data.draw(st.integers(min_value=1, max_value=8), "batch_size")
#     time_steps = data.draw(st.integers(min_value=1, max_value=4), "time_steps")
#     torch.manual_seed(data.draw(st.integers(min_value=0, max_value=10), "torch_seed"))
#     state_dim = data.draw(st.integers(min_value=1, max_value=2), "state_dim")
#
#     network = LinearSystem(state_dim, state_dim)
#     obs = nn.Linear(state_dim, state_dim)
#     ekf = EKFEstimator(
#         EKFCell(
#             network,
#             obs,
#             latent_dim=state_dim,
#             observation_dim=state_dim,
#             is_continuous=is_continuous,
#             initial_variance=np.eye(state_dim),
#         ),
#     )
#
#     z0 = data.draw(tensor_strategy((batch_dim, state_dim)), "z0")
#     ekf_z0 = ekf.cell.get_initial_hidden_state(batch_dim, z0)
#
#     time = torch.linspace(0, 1, time_steps)
#     ys = data.draw(tensor_strategy((time_steps, batch_dim, state_dim)), "observations")
#     ekf_mean, ekf_cov = ekf.forward(time, ys, ekf_z0, return_hidden=True)
#     assert_tensor_symmetric_psd(ekf_cov)
#
#     ekf_mean, ekf_cov = ekf.forward(time, ys, ekf_z0, return_hidden=False)
#     assert_tensor_symmetric_psd(ekf_cov)
#
#
# @given(data=st.data())
# @pytest.mark.parametrize("is_continuous", [True, False])
# @settings(deadline=5000, print_blob=True, max_examples=10)
# def test_EKFEstimator_smooth(data: st.DataObject, is_continuous: bool):
#     batch_dim = data.draw(st.integers(min_value=1, max_value=8), "batch_size")
#     time_steps = data.draw(st.integers(min_value=1, max_value=4), "time_steps")
#     torch.manual_seed(data.draw(st.integers(min_value=0, max_value=10), "torch_seed"))
#     state_dim = data.draw(st.integers(min_value=1, max_value=2), "state_dim")
#
#     network = LinearSystem(state_dim, state_dim)
#     obs = nn.Linear(state_dim, state_dim)
#     ekf = EKFEstimator(
#         EKFCell(
#             network,
#             obs,
#             latent_dim=state_dim,
#             observation_dim=state_dim,
#             is_continuous=is_continuous,
#             initial_variance=np.eye(state_dim),
#         ),
#     )
#
#     z0 = data.draw(tensor_strategy((batch_dim, state_dim)), "z0")
#     ekf_z0 = ekf.cell.get_initial_hidden_state(batch_dim, z0)
#
#     time = torch.linspace(0, 1, time_steps)
#     ys = data.draw(tensor_strategy((time_steps, batch_dim, state_dim)), "observations")
#     ekf_mean, ekf_cov = ekf.forward(time, ys, ekf_z0, return_hidden=True)
#     assert_tensor_symmetric_psd(ekf_cov)
#
#     ekf_mean, ekf_cov = ekf.forward(time, ys, ekf_z0, return_hidden=False)
#     assert_tensor_symmetric_psd(ekf_cov)
