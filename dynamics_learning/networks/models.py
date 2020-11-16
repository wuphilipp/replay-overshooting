from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torch import nn as nn


class MLP(nn.Module):
    """A Multi-layer Perceptron."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layer_sizes: List[int],
        nonlinearity: nn.Module,
        dropout: Optional[float] = None,
        batchnorm: Optional[bool] = False,
    ) -> None:
        """Initialize the MLP. Activations are softpluses.

        Parameters
        ----------
        input_dim : int
            Dimension of the input.
        output_dim : int
            Dimension of the output variable.
        hidden_layer_sizes : List[int]
            List of sizes of all hidden layers.
        nonlinearity : torch.nn.Module
            A the nonlinearity to use (must be a torch module).
        dropout : float
            Dropout probability if applied.
        batchnorm : bool
            Flag for applying batchnorm.
        """
        super(MLP, self).__init__()

        assert type(input_dim) == int
        assert type(output_dim) == int
        assert type(hidden_layer_sizes) == list
        assert all(type(n) is int for n in hidden_layer_sizes)

        # building MLP
        self._mlp = nn.Sequential()
        self._mlp.add_module("fc0", nn.Linear(input_dim, hidden_layer_sizes[0]))
        self._mlp.add_module("act0", nonlinearity)
        if batchnorm:
            self._mlp.add_module("bn0", nn.BatchNorm1d(hidden_layer_sizes[0]))
        if dropout is not None and 0.0 <= dropout and dropout <= 1.0:
            self._mlp.add_module("do0", nn.Dropout(p=dropout))
        for i, (in_size, out_size) in enumerate(
            zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:]), 1
        ):
            self._mlp.add_module(f"fc{i}", nn.Linear(in_size, out_size))
            self._mlp.add_module(f"act{i}", nonlinearity)
            if batchnorm:
                self._mlp.add_module("bn{i}", nn.BatchNorm1d(out_size))
            if dropout is not None and 0.0 <= dropout and dropout <= 1.0:
                self._mlp.add_module("do{i}", nn.Dropout(p=dropout))
        self._mlp.add_module("fcout", nn.Linear(hidden_layer_sizes[-1], output_dim))

        # weight initialization
        for m in self._mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape=(..., in_dim)
            Dimension of the input.
        cond : Optional[torch.Tensor], shape=(..., C), default=None
            Any conditional context.

        Returns
        -------
        mlp_out : torch.Tensor, shape=(..., out_dim)
            Output tensor.
        """
        # checking for conditional context
        if cond is None:
            cond = x.new_tensor(np.zeros(x.shape[:-1] + (0,)))

        return self._mlp(torch.cat([x, cond], dim=-1))


@dataclass
class MLPConfig:
    """Config dataclass for MLP."""

    input_dim: int
    output_dim: int
    hidden_layer_sizes: List[int]
    nonlinearity: nn.Module
    dropout: Optional[float] = None
    batchnorm: Optional[bool] = None

    def create(self) -> MLP:
        """Create a MLP from the config params."""
        return MLP(
            self.input_dim,
            self.output_dim,
            self.hidden_layer_sizes,
            self.nonlinearity,
            self.dropout,
            self.batchnorm,
        )


class ODENet(nn.Module, ABC):
    """Abstract class for a neural ODE."""

    def __init__(self) -> None:
        super(ODENet, self).__init__()

    @abstractmethod
    def forward(
        self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """Return the derivative of the state with respect to time.

        Parameters
        ----------
        time : torch.Tensor, shape=(1)
            The current time.
        state : torch.Tensor, shape(B, X)
            The current state.
            B - Batch size
            X - State dim

        Returns
        -------
        torch.tensor (B, X)
            Returns the derivative of the state with respect to time
        """
        raise NotImplementedError


class SimpleODENet(ODENet):
    """Simple ODE net."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: int,
        hidden_units: int,
        nonlinearity: Any,
        skip: bool = False,
        batchnorm: bool = False,
    ) -> None:
        """Initialize simple ODE net."""
        super(SimpleODENet, self).__init__()

        layers = [
            nn.Linear(input_dim, hidden_units),
            nonlinearity,
        ]

        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nonlinearity)
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_units))

        layers.append(nn.Linear(hidden_units, output_dim))

        self._net = nn.Sequential(*layers)

        # weight initialization
        for m in self._net.modules():
            if isinstance(m, nn.Linear):
                # NOTE: fan_out or fan_in could be good. TODO: make a decision later.
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

        self._skip = skip

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        t : torch.Tensor, shape=(1)
            Current time.
        x : torch.Tensor, shape=(B, n)
            State.
        u : Optional[torch.Tensor], shape=(B, m), default=None
            Control inputs.
        cond : Optional[torch.Tensor], shape=(B, C), default=None
            Any conditional context.
        """
        # check for control input
        if u is None:
            u = x.new_tensor(np.zeros(x.shape[:-1] + (0,)))

        # check for conditional context
        if cond is None:
            cond = x.new_tensor(np.zeros(x.shape[:-1] + (0,)))

        # compute the forward dynamics
        if self._skip:
            return x + self._net(torch.cat([x, u, cond], dim=-1))
        else:
            return self._net(torch.cat([x, u, cond], dim=-1))


@dataclass
class SimpleODENetConfig:
    """Config dataclass for simple ODE net."""

    input_dim: int
    output_dim: int
    hidden_layers: int
    hidden_units: int
    nonlinearity: nn.Module
    skip: bool = False
    batchnorm: bool = False

    def create(self) -> SimpleODENet:
        """Create a simple ODE net from the config params."""
        return SimpleODENet(
            self.input_dim,
            self.output_dim,
            self.hidden_layers,
            self.hidden_units,
            self.nonlinearity,
            self.skip,
            self.batchnorm,
        )


# class SplitRNN(ODENet):
#     """[DEP] Candidate alternative architecture. Doesn't work well right now."""

#     def __init__(
#         self,
#         mlp_state_dim: int,
#         total_state_dim: int,
#         hidden_layers: int,
#         hidden_units: int,
#         nonlinearity: Any,
#     ) -> None:
#         """Create a split RNN.

#         Parameters
#         ----------
#         mlp_state_dim : int
#             Dimension of the state passed through the MLP.
#         total_state_dim : int
#             Total state dimension.
#         hidden_layers : int
#             Number of hidden layers in the MLP.
#         hidden_units : int
#             Number of hidden units in the MLP.
#         nonlinearity : Any
#             Which nonlinearity to use.
#         """
#         super(SplitRNN, self).__init__()
#         self._mlp_dim = mlp_state_dim
#         self._rnn_dim = total_state_dim - mlp_state_dim
#         self._rnn = nn.GRUCell(mlp_state_dim, total_state_dim - mlp_state_dim)
#         self._mlp = MLP(
#             total_state_dim, mlp_state_dim, [hidden_units] * hidden_layers, nonlinearity
#         )

#     def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
#         """See parent class."""
#         x1, x2 = torch.split(x, [self._rnn_dim, self._mlp_dim], dim=-1)

#         x_mlp = self._mlp(x)
#         new_x2 = x_mlp + x2

#         new_x1 = self._rnn(x_mlp, x1)
#         return torch.cat([new_x1, new_x2], dim=-1)


# @dataclass
# class ForwardMlpRnnConfig:
#     """Config dataclass for forward MLP RNN."""

#     mlp_state_dim: int
#     total_state_dim: int
#     hidden_layers: int
#     hidden_units: int
#     nonlinearity: Any

#     def create(self) -> SplitRNN:
#         """Create a forward MLP RNN from the config params."""
#         return SplitRNN(
#             self.mlp_state_dim,
#             self.total_state_dim,
#             self.hidden_layers,
#             self.hidden_units,
#             self.nonlinearity,
#         )


class SimpleLSTM(nn.Module):
    """Simple LSTM."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_lstm_layers: int,
        num_fc_layers: Optional[int],
        hidden_units: int,
        nonlinearity: nn.Module,
        reverse: bool = False,
    ) -> None:
        """Initialize a LSTM.

        Parameters
        ----------
        in_dim : int
            Dimension of the input.
        out_dim : int
            Dimension of the output variable.
        num_layers : int
            Numer of LSTM layers.
        """
        super(SimpleLSTM, self).__init__()

        # init lstm
        self._lstm = nn.LSTM(input_dim, hidden_units, num_lstm_layers)

        # init MLP for prediction if specified
        self._net: Optional[MLP] = None
        if num_fc_layers is not None and num_fc_layers > 0:
            self._net = MLP(
                hidden_units, output_dim, [hidden_units] * num_fc_layers, nonlinearity
            )

        self._hidden_dim = hidden_units
        self._num_lstm_layers = num_lstm_layers
        self._reverse = reverse

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create zeros for hidden states for passing into the LSTM.

        Parameters
        ----------
        batch_size : int
            Batch size.
        device : torch.device
            Device to initalize on.

        Returns
        -------
        torch.Tensor
            Zeroed out hidden state.
        torch.Tensor
            Zeroed out cell state.
        """
        return (
            torch.zeros(
                self._num_lstm_layers, batch_size, self._hidden_dim, device=device
            ),
            torch.zeros(
                self._num_lstm_layers, batch_size, self._hidden_dim, device=device
            ),
        )

    def forward(self, x: torch.Tensor, h_c=None, return_hidden=False) -> torch.Tensor:
        """Return the derivative of the state with respect to time.

        Parameters
        ----------
        state : torch.Tensor, shape(T, B, X)
            The current state.
            B - Batch size
            T - Length of the trajectory.
            X - State dim
        h_c : Tuple[torch.Tensor, torch.Tensor]
            The hidden states of the LSTM.

        Returns
        -------
        torch.tensor (T, B, X)
            Prediction of the next time step.
        """
        _, B, X = x.shape
        if h_c is None:
            hidden, h_c_out = self._lstm(x)
        else:
            hidden, h_c_out = self._lstm(x, h_c)

        if self._reverse:
            x = x.flip(0)

        if return_hidden:
            # TODO fix typing
            if self._net is not None:
                return x + self._net(hidden.permute(1, 0, 2)).permute(1, 0, 2), h_c_out  # type: ignore
            else:
                return x + hidden, h_c_out  # type: ignore
        else:
            if self._net is not None:
                return x + self._net(hidden.permute(1, 0, 2)).permute(1, 0, 2)
            else:
                return x + hidden

    def predict(
        self,
        x0: torch.Tensor,
        steps: int,
        h_c: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Predict outputs over some horizon.

        Parameters
        ----------
        state : torch.Tensor, shape(B, n)
            The current state.
            B - Batch size
            n - State dim
        steps : int
            Number of steps to predict forward
        h_c : Tuple[torch.Tensor, torch.Tensor]
            The hidden states of the LSTM.

        Returns
        -------
        torch.tensor (T, B, n)
            Prediction of the next time step.
        """
        B = x0.shape[-2]
        if h_c is None:
            h_c = self.init_hidden(B, device=x0.device)

        x = x0
        xs = [x]
        for _ in range(steps - 1):
            x, h_c = self(x.unsqueeze(0), h_c, return_hidden=True)
            x = x.squeeze(0)
            xs.append(x)
        return torch.stack(xs)


@dataclass
class SimpleLSTMConfig:
    """Config dataclass for simple LSTM."""

    input_dim: int
    output_dim: int
    num_lstm_layers: int
    num_fc_layers: Optional[int]
    hidden_units: int
    nonlinearity: nn.Module
    reverse: bool = False

    def create(self) -> SimpleLSTM:
        """Create a simple LSTM from the config params."""
        return SimpleLSTM(
            self.input_dim,
            self.output_dim,
            self.num_lstm_layers,
            self.num_fc_layers,
            self.hidden_units,
            self.nonlinearity,
        )
