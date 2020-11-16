from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from torchdiffeq import odeint

from dynamics_learning.networks.models import ODENet


# Dynamic Systems
class DynamicSystem(ODENet, ABC):
    """Template for a DynamicSystem."""

    def __init__(self) -> None:
        super(DynamicSystem, self).__init__()

    def forward(
        self, time: torch.Tensor, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Returns the derivative of the state with respect to time.

        Parameters
        ----------
        time : torch.Tensor, shape=(1)
            The current time.
        x : torch.Tensor, shape=(B, X)
            The current state.
            B - Batch size
            X - State dim
        u : torch.Tensor, shape=(B, m)
            Control input.
            m - Control dimension

        Returns
        -------
        torch.tensor, shape=(B, X)
            Returns the derivative of the state with respect to time
        """
        ctrl = None if u is None or u.shape[-1] == 0 else u
        return self._dx(time, x, u=ctrl)

    @abstractmethod
    def _dx(
        self,
        time: torch.Tensor,
        state: torch.Tensor,
        p: Optional[np.ndarray] = None,
        u: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns the batched derivative of the state with respect to time.

        Parameters
        ----------
        time : torch.Tensor, shape=(1)
            The current time.
        state : torch.Tensor, shape=(B, X)
            The current state.
        p : Optional[np.ndarray], shape=(?)
            Any optional parameters.
        u : Optional[torch.Tensor], shape=(B, M)
            Control inputs.

        Returns
        -------
        torch.tensor, shape=(B, X)
            Returns the derivative of the state with respect to time
        """

    def solve_torch(
        self, time: torch.Tensor, x0: torch.Tensor, method="dopri5"
    ) -> torch.Tensor:
        """Return the solution of the initial value problem. Supports batch solving.

        Parameters
        ----------
        time : torch.Tensor, shape=(T)
            List of times to evaluate the trajectory.
        x0: torch.Tensor, shape=(B, X)
            The inital condition.

        Returns
        -------
        torch.tensor, shape=(T, B, X)
            Returns the derivative of the state with respect to time
        """
        return odeint(self, x0, time, method=method)


class LinearSystem(DynamicSystem):
    """Ground truth dynamics for a linear system.

    The state is can be of any dimension.
    """

    def __init__(self, A: np.ndarray, B: Optional[np.ndarray] = None) -> None:
        """Create a linear system.

        Parameters
        ----------
        A : np.ndarray, shape=(X, X)
            A state transition matrix.
        B : np.ndarray, shape=(X, m)
            Control matrix.
        """
        super(LinearSystem, self).__init__()
        assert len(A.shape) == 2, "Linear system must be two dimensional"
        assert A.shape[0] == A.shape[1], "Linear system must be square"
        self.A = torch.tensor(A, dtype=torch.float).unsqueeze(0)
        self._state_size = self.A.shape[1]
        self.B: Optional[torch.Tensor]
        if B is not None:
            self.B = torch.tensor(B, dtype=torch.float).unsqueeze(0)
        else:
            self.B = None

    def _dx(
        self,
        time: torch.Tensor,
        state: torch.Tensor,
        p: Optional[np.ndarray] = None,
        u: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = (self.A @ state.unsqueeze(-1)).squeeze(-1)
        if self.B is not None and u is not None:
            output = output + (self.B @ u.unsqueeze(-1)).squeeze(-1)
        return output


class VanDerPol(DynamicSystem):
    """Ground truth Van Der Pol dynamics.

    The state is two dimensional and represents position and velocity.
    """

    def __init__(self, mu: float = 1.0) -> None:
        """Create a VanDerPol system.

        Parameters
        ----------
        mu : float
            The mu parameters in the Van Der Pol system.
        """
        super(VanDerPol, self).__init__()
        self._mu = mu

    def _dx(
        self,
        time: torch.Tensor,
        state: torch.Tensor,
        p: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Batched time derivative of the VDP state. See docstring for dx()."""
        assert u is None

        x = state[..., 0:1]
        y = state[..., 1:2]

        if p is None:
            mu = self._mu
        else:
            mu = np.abs(p[0])  # in case of negative samples

        x_dot = mu * (x - 1 / 3 * x ** 3 - y)
        y_dot = 1 / mu * (x)
        return torch.cat([x_dot, y_dot], dim=-1)

    def dx(
        self,
        state: np.ndarray,
        time: np.ndarray,
        p: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Time derivative of VDP state. Used for sdeint.

        Parameters
        ----------
        state : np.ndarray, shape=(2)
            The 2D state of the system.
        time : np.ndarray, shape=(1)
            The current time.
        p : Optional[np.ndarray]
            Extra parameters to pass into the system.
        u : Optional[torch.Tensor], shape=N/A
            Must be None. Control not supported.

        Returns
        -------
        dx : np.ndarray, shape=(2)
            Time derivative of the state.
        """
        assert u is None

        x = state[0:1]
        y = state[1:2]

        if p is None:
            mu = self._mu
        else:
            mu = np.abs(p[0])  # in case of negative samples

        x_dot = mu * (x - 1 / 3 * x ** 3 - y)
        y_dot = 1 / mu * (x)
        return np.concatenate((x_dot, y_dot))


class Pendulum(DynamicSystem):
    """Ground dynamics for a pendulum with friction.

    The state is two dimensional and represents angular position and velocity.
    """

    def __init__(
        self,
        mass: float = 1.0,
        length: float = 1.0,
        friction: float = 0.1,
        gravity: float = 9.80665,
    ) -> None:
        """Initialize a pendulum system.

        Parameters
        ----------
        mass : float, default=1.0
            The mass of the pendulum.
        length : float, default=1.0
            The length of the pendulum.
        friction : float, default=0.1
            The friction coefficient.
        gravity : float, default=9.80665
            Gravity.
        """
        super(Pendulum, self).__init__()
        self._mass = mass
        self._length = length
        self._friction = friction
        self._g = gravity

    def _dx(
        self,
        time: torch.Tensor,
        state: torch.Tensor,
        p: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Batched time derivative of the pendulum state. See docstring for dx()."""
        x1 = state[..., 0:1]
        x2 = state[..., 1:2]

        if p is None:
            mass = self._mass
            length = self._length
            friction = self._friction
        else:
            assert len(p) == 3
            mass = p[0]
            length = p[1]
            friction = p[2]
        g = self._g

        if u is None:
            torque = 0.0
        else:
            torque = u

        dx1 = x2
        dx2 = -(g / length) * torch.sin(x1) - (friction / mass) * x2 + torque
        return torch.cat([dx1, dx2], dim=-1)

    def dx(
        self,
        state: np.ndarray,
        time: np.ndarray,
        p: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Time derivative of pendulum state. Used for sdeint.

        Parameters
        ----------
        state : np.ndarray, shape=(2)
            The 2D state of the system.
        time : np.ndarray, shape=(1)
            The current time.
        p : Optional[np.ndarray]
            Extra parameters to pass into the system.
        u : Optional[np.ndarray], shape=(1)
            Input torque.

        Returns
        -------
        dx : np.ndarray, shape=(2)
            Time derivative of the state.
        """
        x1 = state[0:1]
        x2 = state[1:2]

        if p is None:
            mass = self._mass
            length = self._length
            friction = self._friction
        else:
            assert len(p) == 3
            mass = p[0]
            length = p[1]
            friction = p[2]
        g = self._g

        if u is None:
            torque = np.array([0.0])
        else:
            torque = u

        dx1 = x2
        dx2 = -(g / length) * np.sin(x1) - (friction / mass) * x2 + torque
        return np.concatenate((dx1, dx2))


class Lorenz(DynamicSystem):
    """Ground truth Van Der Pol dynamics.

    The state is three-dimensional.
    """

    def __init__(self):
        super(Lorenz, self).__init__()
        # TODO allow for parameter inputs

    def _dx(
        self,
        time: torch.Tensor,
        state: torch.Tensor,
        p: Optional[np.ndarray] = None,
        u: Optional[torch.Tensor] = None,
    ) -> np.array:
        x1 = state[..., 0:1]
        x2 = state[..., 1:2]
        x3 = state[..., 2:3]

        dx1 = 10.0 * (x2 - x1)
        dx2 = x1 * (28 - x3) - x2
        dx3 = x1 * x2 - 8 / 3 * x3
        return torch.cat([dx1, dx2, dx3], dim=-1)


if __name__ == "__main__":
    """A quick demo."""
    x0 = torch.tensor([[5.5, 1.1], [-5.5, 1.1], [0.5, 0.1]], dtype=torch.float).reshape(
        3, 2
    )
    sys = VanDerPol(mu=1.1)
    t = torch.linspace(0.0, 10, 100)
