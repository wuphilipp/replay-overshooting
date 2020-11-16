import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from dynamics_learning.networks.dynamics import (
    DynamicSystem,
    LinearSystem,
    Lorenz,
    Pendulum,
    VanDerPol,
)


@st.composite
def square_array_strategy(draw):
    """Generate a square np.array. Shape is (N, N)."""
    shape_side = draw(st.integers(min_value=1, max_value=32))
    A = hnp.arrays(
        dtype=np.float32,
        shape=(shape_side, shape_side),
        elements=st.floats(min_value=-1e3, max_value=1e3),
    )
    return A


@st.composite
def linear_system_strategy(draw):
    return LinearSystem(A=draw(square_array_strategy()))


@st.composite
def dynamic_system_strategy(draw):
    system_list = [
        st.just(LinearSystem),
        st.just(Pendulum),
        st.just(VanDerPol),
        st.just(Lorenz),
    ]
    return draw(st.one_of(system_list))


@given(sys=dynamic_system_strategy())
def test_dynamics_system_strategy(sys):
    assert issubclass(sys, DynamicSystem)


def test_LinearSystem():
    y0 = torch.tensor([3.1, 4.1], dtype=torch.float).reshape(1, 2)
    t = torch.linspace(0.0, 25.0, 10)
    A = np.array([[-10.5, 0], [0, -0.1]])
    sys = LinearSystem(A)
    sys.solve_torch(t, y0)


def test_VanDerPol():
    y0 = torch.tensor([3.1, 4.1], dtype=torch.float).reshape(1, 2)
    sys = VanDerPol(mu=1.1)
    t = torch.linspace(0.0, 25.0, 10)
    sys.solve_torch(t, y0)


def test_Pendulum():
    y0 = torch.tensor([3.1, 4.1], dtype=torch.float).reshape(1, 2)
    sys = Pendulum(1.1)
    t = torch.linspace(0.0, 25.0, 10)
    sys.solve_torch(t, y0)
