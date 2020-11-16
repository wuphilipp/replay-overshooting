import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from fannypack.utils import Buddy
from matplotlib import pyplot as plt
from torch import nn as nn

from dynamics_learning.data.datasets import DatasetConfig, VisData
from dynamics_learning.utils.net_utils import gaussian_log_prob, reparameterize_gauss
from dynamics_learning.utils.plot_utils import PlotHandler as ph


class Estimator(nn.Module, ABC):
    """Estimates the state and observations."""

    @property
    def config(self) -> "EstimatorConfig":
        """Estimator config."""
        return self._config

    def __init__(self, config: "EstimatorConfig") -> None:
        """Initialize the estimator."""
        super(Estimator, self).__init__()
        self._config = config

    @abstractmethod
    def forward(
        self,
        time: torch.Tensor,
        y: torch.Tensor,
        u: torch.Tensor,
        z0: torch.Tensor,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the filter rollout.

        Parameters
        ----------
        time : torch.Tensor, shape=(T)
            The current time.
        y : torch.Tensor, shape=(T, B, p)
            The states that correspond to all time except the last one.
            B - Batch size
            p - Observation dim
        u : torch.Tensor, shape=(T, B, m)
            The control inputs.
            m - Control dim
        z0 : torch.Tensor, shape=(B, n)
            The initial hidden state.
            n - Latent state dim
        return_hidden : bool, default=False
            Return the mean and covariance of the hidden state at each time step instead.

        Returns
        -------
        mean : torch.tensor, shape=(T, B, p) or (T, B, n) if return_hidden=True
            Returns the predicted mean.
        covariance : torch.tensor, shape=(T, B, p, p) or (T, B, n, n) if return_hidden=True
            Returns the predicted covariance.
        """

    @abstractmethod
    def loss(
        self,
        batch_t: torch.Tensor,
        batch_y: torch.Tensor,
        batch_u: torch.Tensor,
        iteration: int,
        avg: bool,
    ) -> torch.Tensor:
        """Compute the loss.

        Parameters
        ----------
        batch_t : torch.Tensor, shape=(T)
            Times.
        batch_y : torch.Tensor, shape=(T, B, p)
            Data.
        batch_u : torch.Tensor, shape=(T, B, m)
            Control inputs.
        iteration : int
            The current training iteration (used for things like scheduling).
        avg : bool, default=True
            Flag indicating whether to average the loss.
        """

    @abstractmethod
    def predict(
        self,
        z0_mu: torch.Tensor,
        z0_cov: torch.Tensor,
        pred_times: torch.Tensor,
        u: torch.Tensor,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward inference method.

        Parameters
        ----------
        z0_mu : torch.Tensor, shape=(B, n)
            Batch of initial mean hidden state.
        z0_cov : torch.Tensor, shape=(B, n, n)
            Batch of initial covariances of the hidden state.
        pred_times : torch.Tensor, shape=(T)
            Array of observation waypoint times.
        u : torch.Tensor
            Control inputs.
        return_hidden : bool, default=False
            Return the mean and covariance of the hidden state at each time step instead.

        Returns
        -------
        mean : torch.Tensor, shape=(T, B, p) or (T, B, n) if return_hidden=True
            The predicted mean observations.
        covariance : torch.Tensor, shape=(T, B, p, p) or (T, B, n, n) if return_hidden=True
            The predicted covariances.
        """

    @abstractmethod
    def get_initial_hidden_state(
        self, batch_size: int, z0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return a nominal initial hidden state.

        Parameters
        ----------
        batch_size : int
            Batch size
        z0 : Optional[torch.Tensor], default=None
            A prespecified intial hidden state. By default one is initalize automatically.

        Returns
        -------
        torch.Tensor
            Batched initial hidden state
        """

    @abstractmethod
    def latent_to_observation(
        self, z_mu: torch.Tensor, z_cov: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes a sequence of latent variables into reconstructed data.

        Parameters
        ----------
        z_mu : torch.Tensor, shape=(..., n)
            Batch of initial mean hidden state.
        z_cov : torch.Tensor, shape=(..., n, n)
            Batch of initial covariances of the hidden state.

        Returns
        -------
        y_mean : torch.Tensor, shape=(..., p)
            The predicted mean observations.
        y_cov : torch.Tensor, shape=(..., p, p)
            The predicted covariances.
        """

    @abstractmethod
    def log(self, buddy: Buddy, viz: VisData, filter_length: int = 1) -> None:
        """Logs data during training.

        Parameters
        ----------
        buddy : Buddy
            A buddy.
        viz : VisData
            Visualization data.
        filter_length : int
            The number of datapoints to filter on.
        """

    def eval_loss(self, viz: VisData, filt_points: int, pred_points: int) -> None:
        """Prints evaluation losses for the model.

        This is the default implementation suitable for Gaussian estimators.

        Parameters
        ----------
        viz : VisData
            The visualization data with which to compute the loss.
        filt_points : int
            The number of points with which to filter.
        pred_points : int
            The desired number of prediction points.
        """
        assert filt_points + pred_points <= len(viz.t)

        t = viz.t
        y_data = viz.y
        u = viz.u
        B = y_data.shape[1]

        # filtering and prediction time/data
        t_filt = t[:filt_points]
        y_filt = y_data[:filt_points]
        u_filt = u[:filt_points]
        t_pred = t[(filt_points - 1) : (filt_points + pred_points - 1)]
        y_pred = y_data[(filt_points - 1) : (filt_points + pred_points - 1)]
        u_pred = u[(filt_points - 1) : (filt_points + pred_points - 1)]

        # filtering
        z0_f = self.get_initial_hidden_state(B)
        z_mu_f, z_cov_f = self(t_filt, y_filt, u_filt, z0_f, return_hidden=True)

        # prediction
        z0_mu_p = z_mu_f[-1]
        z0_cov_p = z_cov_f[-1]
        y_mu_p, y_cov_p = self.predict(z0_mu_p, z0_cov_p, t_pred, u_pred)

        # computing losses (NLL and L2)
        loss_nll = -torch.mean(gaussian_log_prob(y_mu_p, y_cov_p, y_pred))
        y_samps = torch.stack(
            [reparameterize_gauss(y_mu_p, y_cov_p) for i in range(100)]
        )
        loss_ade = torch.mean(torch.sqrt((y_samps - y_pred.unsqueeze(0)) ** 2))  # ADE

        # reporting the evaluation
        print(
            f"Prediction Loss (filt_pts={filt_points}, pred_pts={pred_points}) \t"
            f"NLL Loss: {loss_nll.item():.3f} \t ADE Loss: {loss_ade.item():.5f}"
        )

    def summary_plot(self, viz: VisData, name: str, debug: bool = False) -> None:
        """Produces and saves summary plots.

        Parameters
        ----------
        viz : VisData
            The visualization data with which to compute the loss.
        name : str
            The name of the experiment.
        debug : bool, default=False
            Flag indicating whether debug is on.
        """
        assert viz.y.shape[-1] == 2
        assert len(viz.t) >= 30

        # filter on 10 points, predict on 20 points
        filt_points = 10
        pred_points = 20
        t = viz.t
        y_data = viz.y
        u = viz.u
        B = y_data.shape[1]
        idx = np.random.randint(B)  # choose random traj to visualize

        # filtering and prediction time/data
        t_filt = t[:filt_points]
        y_filt = y_data[:filt_points, idx : idx + 1]
        u_filt = u[:filt_points, idx : idx + 1]

        start_idx = filt_points - 1
        end_idx = filt_points + pred_points - 1
        t_pred = t[start_idx:end_idx]
        y_pred = y_data[start_idx:end_idx, idx : idx + 1]  # slice, preserve B dim
        u_pred = u[start_idx:end_idx, idx : idx + 1]

        # filtering
        z0_f = self.get_initial_hidden_state(B)[idx : idx + 1]

        # TODO: when return_hidden, probably return 4 things instead of needing to
        # call this function twice since it computes the exact same things
        z_mu_f, z_cov_f = self(t_filt, y_filt, u_filt, z0_f, return_hidden=True)
        y_mu_f, y_cov_f = self(t_filt, y_filt, u_filt, z0_f)

        # prediction
        z0_mu_p = z_mu_f[-1]
        z0_cov_p = z_cov_f[-1]
        y_mu_p, y_cov_p = self.predict(z0_mu_p, z0_cov_p, t_pred, u_pred)

        # making plotting directory
        if debug:
            plt.show()
        if not os.path.exists("summary_plots"):
            os.mkdir("summary_plots")

        # plotting the loss per time step
        # TODO: replace the loss functions here later if we want this reported
        _losses = self.loss(t_pred, y_pred, u_pred, int(1e9), avg=False)
        losses = (torch.sum(_losses, dim=1) / B).cpu().numpy()  # (T, 1)
        fig1 = plt.figure()
        plt.plot(losses)
        plt.savefig("summary_plots/summary_plots-" + name + "-loss_step.png")
        plt.close(fig1)

        # generating the plots
        fig2 = plt.figure(figsize=(6, 3))  # (width, height)

        # filtered/predicted means vs. data
        ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)
        ph.plot_xy_compare(
            [y_mu_p.numpy(), y_pred.numpy(), y_mu_f.numpy(), y_filt.numpy()],
            style_list=["r-", "b-", "r:", "b:"],
            startmark_list=[None, None, "ro", "bo"],
            endmark_list=["rx", "bx", None, None],
        )
        ax1.set_title("Filtered/Predicted Means vs. Data", fontsize=10)
        plt.axis("square")

        # X timeseries
        ax2 = plt.subplot2grid((2, 4), (0, 2), colspan=2)
        ph.plot_traj_with_var(
            t_filt.numpy(),
            y_mu_f[..., 0, 0],
            y_cov_f[..., 0, 0, 0],
            linestyle=":",
            color="r",
        )
        ph.plot_traj(t_filt.numpy(), y_filt[..., 0], linestyle=":", color="b")
        ph.plot_traj_with_var(
            t_pred.numpy(),
            y_mu_p[..., 0, 0],
            y_cov_p[..., 0, 0, 0],
            linestyle="-",
            color="r",
        )
        ph.plot_traj(t_pred.numpy(), y_pred[..., 0], linestyle="-", color="b")
        ax2.set_title("XY Timeseries with Uncertainty", fontsize=10)
        ax2.set_ylabel("x")

        # Y timeseries
        ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
        ph.plot_traj_with_var(
            t_filt.numpy(),
            y_mu_f[..., 0, 1],
            y_cov_f[..., 0, 1, 1],
            linestyle=":",
            color="r",
        )
        ph.plot_traj(t_filt.numpy(), y_filt[..., 1], linestyle=":", color="b")
        ph.plot_traj_with_var(
            t_pred.numpy(),
            y_mu_p[..., 0, 1],
            y_cov_p[..., 0, 1, 1],
            linestyle="-",
            color="r",
        )
        ph.plot_traj(t_pred.numpy(), y_pred[..., 1], linestyle="-", color="b")
        ax3.set_ylabel("y")
        ax3.set_xlabel("t")

        # saving plots
        plt.tight_layout()
        plt.savefig("summary_plots/summary_plots-" + name + "-timeseries.png")
        plt.close(fig2)


@dataclass(frozen=True)
class EstimatorConfig:
    """Estimator specific configuration parameters.

    Parameters
    ----------
    latent_dim : int
        Dimension of the state.
    ctrl_dim : int
        Dimension of the control input.
    dataset : DatasetType
        The dataset type.
    """

    latent_dim: int
    ctrl_dim: int
    dataset: DatasetConfig

    def create(self) -> Estimator:
        """Create an estimator from its config class.

        Returns
        -------
        Estimator
            An `Estimator`.
        """
        raise NotImplementedError
