from typing import Callable, Tuple, Union

import numpy as np
import torch
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy import testing as npt
from torch import nn as nn

from dynamics_learning.utils.net_utils import (
    batch_eye,
    gaussian_log_prob,
    jacobian,
    lower_matrix_to_vector,
    lower_vector_to_matrix,
    quadratic_matmul,
    safe_chol,
)


def assert_allclose_tensors(
    t1: torch.Tensor, t2: torch.Tensor, rtol=1e-5, atol=1e-5
) -> None:
    """Check that a pair of torch.Tensor's are equal.

    t1: torch.Tensor
        Tensor 1.
    t2: torch.Tensor
        Tensor 2.
    """
    assert t1.shape == t2.shape
    npt.assert_allclose(t1.detach().numpy(), t2.detach().numpy(), rtol=rtol, atol=atol)


@st.composite
def tensor_strategy(
    draw,
    shape: Union[int, Tuple[int, ...], st.SearchStrategy[Tuple[int, ...]]],
    min_value: float = -1e2,
    max_value: float = 1e2,
) -> torch.Tensor:
    tensor_element_strategy = st.floats(
        min_value=min_value, max_value=max_value, width=32
    )
    x = draw(hnp.arrays(shape=shape, dtype=np.float, elements=tensor_element_strategy))
    return torch.tensor(x, dtype=torch.float)


def wrap_zero_input(
    module: nn.Module,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Wrap a function call that requires a second input.

    This helper is for making older test functions compatible
    with the new kf cell interface.

    Parameters
    ----------
    module: nn.Module
        The torch module.

    Returns
    -------
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        A callable function that odeint can consume.
    """
    return lambda t, z: module(t, z, torch.zeros(z.shape[:-1] + (0,)))


@given(x=tensor_strategy(shape=hnp.array_shapes(min_dims=1, max_dims=1)))
def test_quat_mul_simple(x: torch.Tensor):
    eye = torch.eye(x.shape[-1])
    assert_allclose_tensors(x.dot(x), quadratic_matmul(x, eye))


@given(x=tensor_strategy(shape=hnp.array_shapes(min_dims=1, max_dims=10)))
def test_quat_mul_simple_batch(x: torch.Tensor):
    eye = torch.eye(x.shape[-1]).expand(x.shape + (x.shape[-1],))

    manual_dot = torch.sum(x * x, dim=-1)
    assert_allclose_tensors(manual_dot, quadratic_matmul(x, eye))


@given(
    B=st.integers(min_value=1, max_value=100), X=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=10)  # this is a simple function
def test_batch_eye(B: int, X: int):
    b_eye = batch_eye(X, B)

    assert b_eye.shape == (B, X, X)

    for b in range(B):
        assert (b_eye[b, :, :] == torch.eye(X)).all()


@given(
    cov_matrix=tensor_strategy(shape=(16, 5, 5)),
    mean_vector=tensor_strategy(shape=(16, 5)),
    x=tensor_strategy(shape=(16, 5)),
    use_torch=st.booleans(),
)
def test_gaussian_log_prob(
    cov_matrix: torch.Tensor,
    mean_vector: torch.Tensor,
    x: torch.Tensor,
    use_torch: bool,
):
    cov = cov_matrix @ cov_matrix.transpose(-1, -2) + torch.eye(5)
    distribution = torch.distributions.multivariate_normal.MultivariateNormal(
        mean_vector, cov
    )
    torch_result = distribution.log_prob(x)
    assert_allclose_tensors(
        torch_result, gaussian_log_prob(mean_vector, cov, x, use_torch), rtol=1e-2
    )


@given(
    cov_matrix=tensor_strategy(shape=(8, 16, 5, 5)),
    mean_vector=tensor_strategy(shape=(8, 16, 5)),
    x=tensor_strategy(shape=(8, 16, 5)),
)
def test_gaussian_log_prob_more_dims(
    cov_matrix: torch.Tensor, mean_vector: torch.Tensor, x: torch.Tensor
):
    cov = cov_matrix @ cov_matrix.transpose(-1, -2) + torch.eye(5)
    distribution = torch.distributions.multivariate_normal.MultivariateNormal(
        mean_vector, cov
    )
    torch_result = distribution.log_prob(x)
    assert_allclose_tensors(
        torch_result, gaussian_log_prob(mean_vector, cov, x), rtol=1e-2
    )


@given(x=tensor_strategy(shape=(1, 3)),)
def test_jacobian(x: torch.Tensor):
    class TestModule(nn.Module):
        """The state can be of any dimension."""

        def __init__(self):
            """Create a linear system."""
            super(TestModule, self).__init__()
            self.fc = torch.nn.Linear(3, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    f = TestModule()
    J = jacobian(f, x, 4)

    A = [p for p in f.parameters()][0]  # parameters of matrix
    assert_allclose_tensors(A, J.squeeze(0))


@given(x=tensor_strategy(shape=(1, 3)), u=tensor_strategy(shape=(1, 2)))
def test_jacobian_input(x: torch.Tensor, u: torch.Tensor):
    class TestModule(nn.Module):
        """The state can be of any dimension."""

        def __init__(self) -> None:
            """Create a linear system."""
            super(TestModule, self).__init__()
            self.fc = torch.nn.Linear(3 + 2, 4)

        def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            net_input = torch.cat([x, u], dim=-1)
            return self.fc(net_input)

    def jacobian_input(
        f: nn.Module, z: torch.Tensor, out_dim: int, u: torch.Tensor
    ) -> torch.Tensor:
        """Compute the batched Jacobian for this specific test case."""
        z_jac = z.unsqueeze(1).repeat(1, out_dim, 1)  # (B, out_dim, n)
        z_jac.requires_grad_(True)
        if u is not None and u.shape[-1] > 0:
            u_jac = u.unsqueeze(1).repeat(1, out_dim, 1)  # (B, out_dim, m)
            u_jac = u_jac.reshape(-1, 2)
        else:
            u_jac = torch.Tensor(*z_jac.shape[:-1], 0)
        y = f(z_jac.reshape(-1, 3), u_jac).reshape(1, -1, out_dim)
        mask = batch_eye(out_dim, 1).to(z.device)
        jac = torch.autograd.grad(y, z_jac, mask, create_graph=True)
        return jac[0]

    f = TestModule()
    J = jacobian_input(f, x, 4, u)

    A = [p for p in f.parameters()][0]  # parameters of matrix
    assert_allclose_tensors(A[:, :3], J.squeeze(0))

    # this code here is a good way to test for nonlinear stuff if we need to
    # --------------------------------------------------------------------------
    # jac_ = torch.autograd.functional.jacobian(
    #     lambda z: self._dynamics(t, z, u=u), z, create_graph=True
    # )  # shape=(B, n, B, n), where the B dims are paired up and rest are 0
    # jac = torch.stack([jac_[i, :, i, :] for i in range(jac_.shape[0])])
    # return jac
    # --------------------------------------------------------------------------


@given(batch_matrix=tensor_strategy(shape=(8, 4, 4)))
def test_lower_matrix_to_vector(batch_matrix: torch.Tensor):
    # batch_matrix is a random matrix of shape(B, X, X)
    sym = batch_matrix @ batch_matrix.transpose(-1, -2) + torch.eye(4)
    lower = torch.cholesky(sym)
    lower_after_transforms = lower_vector_to_matrix(lower_matrix_to_vector(lower))
    assert_allclose_tensors(lower, lower_after_transforms)


@given(batch_matrix=tensor_strategy(shape=(2, 8, 4, 4)))
def test_lower_matrix_to_vector_more_dims(batch_matrix: torch.Tensor):
    # batch_matrix is a random matrix of shape(L, B, X, X)
    sym = batch_matrix @ batch_matrix.transpose(-1, -2) + torch.eye(4)
    print(sym[0, 0, :, :])
    lower = torch.cholesky(sym)
    lower_after_transforms = lower_vector_to_matrix(lower_matrix_to_vector(lower))
    assert_allclose_tensors(lower, lower_after_transforms)


@given(A=tensor_strategy(shape=(8, 4, 4), min_value=-1e10, max_value=1e10))
@example(A=torch.diag_embed(torch.Tensor([1e10, 1, 1])))
@example(A=torch.diag_embed(torch.Tensor([1e10, 1, -1])))
def test_safe_chol(A: torch.Tensor):
    # random symmetric matrix
    A_sym = 0.5 * (A + A.transpose(-1, -2))
    flag = True
    try:
        safe_chol(A_sym)
    except Exception:
        flag = False
        print(torch.symeig(A_sym))
    assert flag

    # test specifically on negative definite matrices
    A_nd = -(A_sym + A_sym.shape[-1] * torch.eye(A_sym.shape[-1]).repeat(8, 1, 1))
    flag = True
    try:
        safe_chol(A_nd)
    except Exception:
        flag = False
        print(torch.symeig(A_nd))
    assert flag
