from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint as odeint

from dynamics_learning.networks.kalman.ekf import (
    EKFCell,
    EKFEstimator,
    EKFEstimatorConfig,
    EKSmoothCache,
)
from dynamics_learning.utils.net_utils import reg_psd, reparameterize_gauss

# TODO: fix the y_hist interactions with the EKF interface to make it not stateful. Also
# clean up all the hacky code that results from a stateful y_hist.

# ------- #
# HELPERS #
# ------- #


class WeightModel(nn.Module):
    """Nonlinear weight model."""

    def __init__(self, num_submodels: int, obs_dim: int, cond_dim: int = 0) -> None:
        """Initialize the weight model."""
        super(WeightModel, self).__init__()

        self._obs_dim = obs_dim
        self._cond_dim = cond_dim

        self._rnn = nn.GRU(obs_dim + cond_dim, num_submodels, 1)
        self._num_submodels = num_submodels

    def forward(
        self,
        y_hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        full_seq: bool = False,
    ) -> torch.Tensor:
        """Takes sequence of observations and outputs weights.

        Parameters
        ----------
        y_hist : torch.Tensor, shape=(T, B, p)
            Sequence of past observations at time T including starting code y0.
        cond : Optional[torch.Tensor], shape=(B, C), default=None
            Conditional context.
        full_seq : bool, default=False
            Flag indicating whether to return weights for an entire sequence.

        Returns
        -------
        weights : torch.Tensor, shape=(B, num_submodels) OR shape=(T, B, num_submodels)
            Batched weights for submodels.
        """
        if cond is None:
            rnn_input = y_hist
        else:
            assert self._cond_dim == cond.shape[-1]
            if len(cond.shape) > 2:
                cond = cond[0]
            T = y_hist.shape[0]
            rnn_input = torch.cat([y_hist, cond.unsqueeze(0).repeat(T, 1, 1)], dim=-1)

        if full_seq:
            d_all, _ = self._rnn(rnn_input)  # returns all hidden states
            weights = F.softmax(d_all[0:-1], dim=-1)
        else:
            _, d = self._rnn(rnn_input)  # returns last hidden state
            weights = F.softmax(d.squeeze(), dim=-1)
        return weights


class DynEnsemble(nn.Module):
    """Dynamics ensemble."""

    def __init__(
        self,
        num_submodels: int,
        latent_dim: int,
        ctrl_dim: int,
        weight_model: WeightModel,
        skip: bool = False,
    ) -> None:
        """Initialize the dynamics ensemble."""
        super(DynEnsemble, self).__init__()
        self._As = nn.Parameter(torch.randn(num_submodels, latent_dim, latent_dim))
        self._Bs = nn.Parameter(torch.randn(num_submodels, latent_dim, ctrl_dim))
        self._weight_model = weight_model
        self._skip = skip

    def get_A(self, alpha: torch.Tensor) -> torch.Tensor:
        """Returns A."""
        return torch.sum(self._As.unsqueeze(0) * alpha[..., None, None], dim=1)

    def get_B(self, alpha: torch.Tensor) -> torch.Tensor:
        """Returns B."""
        return torch.sum(self._Bs.unsqueeze(0) * alpha[..., None, None], dim=1)

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor,
        y_hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward dynamics.

        Parameters
        ----------
        t : torch.Tensor, shape=(1) OR shape=(T)
            Current time.
        z : torch.Tensor, shape=(B, n) OR shape=(T, B, n)
            Current latent state.
        u : torch.Tensor, shape=(B, m) OR shape=(T, B, m)
            Control inputs.
        y_hist : torch.Tensor, shape=(T - 1, B, p)
            Sequence of past observations at time T.
        cond : Optional[torch.Tensor], shape=(B, C), default=None
            Conditional context.

        Returns
        -------
        out : torch.Tensor, shape=(B, n)
            Next latent state (discrete) or time derivative (continuous).
        """
        # determining whether in KL loss function or not
        if z.shape[0] > y_hist.shape[1]:
            alpha = self._weight_model(y_hist, cond=cond, full_seq=True).reshape(
                z.shape[0], -1
            )
        else:
            alpha = self._weight_model(y_hist, cond=cond)
        At = self.get_A(alpha)
        Bt = self.get_B(alpha)

        if Bt.shape[-1] == 0:
            dyn_input = torch.zeros_like(z.unsqueeze(-1))
        else:
            dyn_input = Bt @ u

        if self._skip:
            out = z + (At @ z.unsqueeze(-1) + dyn_input).squeeze(-1)
        else:
            out = (At @ z.unsqueeze(-1) + dyn_input).squeeze(-1)
        return out


class ObsEnsemble(nn.Module):
    """Observation ensemble."""

    def __init__(
        self,
        num_submodels: int,
        latent_dim: int,
        obs_dim: int,
        weight_model: WeightModel,
    ) -> None:
        """Initialize the observation model."""
        super(ObsEnsemble, self).__init__()
        self._Cs = nn.Parameter(torch.randn(num_submodels, obs_dim, latent_dim))
        self._weight_model = weight_model

    def get_C(self, alpha: torch.Tensor, full_seq: bool = False) -> torch.Tensor:
        """Returns C."""
        if full_seq:
            return torch.sum(self._Cs[None, None, ...] * alpha[..., None, None], dim=2)
        else:
            return torch.sum(self._Cs.unsqueeze(0) * alpha[..., None, None], dim=1)

    def forward(
        self,
        z: torch.Tensor,
        y_hist: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Obs model.

        Parameters
        ----------
        z : torch.Tensor, shape=(B, n)
            Current latent state.
        y_hist : torch.Tensor, shape=(T - 1, B, p)
            Sequence of past observations at time T.
        cond : Optional[torch.Tensor], shape=(B, C), default=None
            Conditional context.

        Return
        ------
        y : torch.Tensor, shape=(B, p)
            Current observation.
        """
        alpha = self._weight_model(y_hist, cond=cond)
        Ct = torch.sum(self._Cs.unsqueeze(0) * alpha[..., None, None], dim=1)
        y = (Ct @ z.unsqueeze(-1)).squeeze(-1)
        return y


# ------ #
# LE-EKF #
# ------ #


class LinearEnsembleEKFCell(EKFCell):
    """An ensemble of linear dyn/obs models in a cell.

    See: A Disentangled Recognition and Nonlinear
         Dynamics Model for Unsupervised Learning
    """

    def __init__(
        self,
        num_submodels: int,
        latent_dim: int = 2,
        observation_dim: int = 2,
        ctrl_dim: int = 0,
        cond_dim: int = 0,
        initial_state: Optional[np.ndarray] = None,
        initial_variance: Optional[np.ndarray] = None,
        process_noise: Optional[np.ndarray] = None,
        measurement_noise: Optional[np.ndarray] = None,
        is_continuous: bool = True,
        const_var: bool = False,
        reparam: bool = False,
        regularizer: float = 1e-3,
    ) -> None:
        """See parent class."""
        super(LinearEnsembleEKFCell, self).__init__(
            nn.GRU(1, 1),  # init dummy dynamics model
            nn.GRU(1, 1),  # init dummy obs model
            latent_dim=latent_dim,
            observation_dim=observation_dim,
            ctrl_dim=ctrl_dim,
            initial_state=initial_state,
            initial_variance=initial_variance,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            is_continuous=is_continuous,
            const_var=const_var,
            reparam=reparam,
            regularizer=regularizer,
        )

        # overwrite dyn/obs models with ensemble models
        self._weight_model = WeightModel(
            num_submodels, observation_dim, cond_dim=cond_dim
        )
        self._dynamics: DynEnsemble = DynEnsemble(
            num_submodels,
            latent_dim,
            ctrl_dim,
            self._weight_model,
            skip=not self._is_continuous,
        )
        self._observation_dynamics: ObsEnsemble = ObsEnsemble(
            num_submodels, latent_dim, observation_dim, self._weight_model
        )

        # maintain history of observations. y0 is a learned starting code.
        self.y0 = nn.Parameter(torch.randn(1, 1, self._observation_dim))
        self._y_hist: Optional[torch.Tensor] = None

    # --------- #
    # UTILITIES #
    # --------- #

    def check_y_hist(self) -> bool:
        """Checks whether y_hist is set correctly."""
        return self._y_hist is not None

    def get_y_hist(self) -> torch.Tensor:
        """Gets y_hist."""
        if self._y_hist is None:
            raise RuntimeError("Must set y_hist before getting it!")
        return self._y_hist

    def set_y_hist(self, new_y_hist: torch.Tensor) -> None:
        """Sets y_hist."""
        self._y_hist = new_y_hist

    def clear_y_hist(self, hist_reset_val: Optional[torch.Tensor] = None) -> None:
        """Clear the obs history."""
        if hist_reset_val is None:
            self._y_hist = None
        else:
            self._y_hist = hist_reset_val

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._smooth_cache = EKSmoothCache.create_empty()
        self.clear_y_hist()  # also reset y_hist here for API compliance

    def observation_dynamics(self, z: torch.Tensor, cond=None) -> torch.Tensor:
        """Observe on latent state."""
        return self._observation_dynamics(z, self.get_y_hist(), cond=cond)

    def latent_to_observation(
        self, z_mu: torch.Tensor, z_cov: torch.Tensor, cond=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """See parent class."""
        if self._reparam:
            z_mu = reparameterize_gauss(z_mu, z_cov, device=self._device)
            y_mu = self.observation_dynamics(z_mu, cond=cond)
            shape = y_mu.shape
            return y_mu, self.R.repeat(list(shape[:-1]) + [1, 1])
        else:
            y_mu = self.observation_dynamics(z_mu, cond=cond)
            shape = y_mu.shape
            Z = z_mu.shape[-1]
            Y = y_mu.shape[-1]
            alpha = self._weight_model(self.get_y_hist(), cond=cond, full_seq=True)
            C = self._observation_dynamics.get_C(alpha, full_seq=True).reshape(
                list(shape[:-1]) + [Y, Z]
            )
            y_cov = C @ z_cov @ C.transpose(-1, -2) + self.R.repeat(
                list(shape[:-1]) + [1, 1]
            )
            return y_mu, y_cov

    def _kl_loss_detect(
        self, mu: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Detects whether kl_loss() is calling, returns appropriate weight.

        Parameters
        ----------
        mu : torch.Tensor, shape=(B, n)
            Latent mean.
        cond : Optional[torch.Tensor], shape=(B, C), default=None
            Conditional context.

        Returns
        -------
        alpha : torch.Tensor, shape=(B, num_submodels) OR shape=(T, B, num_submodels)
            Model weight parameters for the linear ensemble.
        """
        if mu.shape[0] > self.get_y_hist().shape[1]:
            alpha = self._weight_model(
                self.get_y_hist(), cond=cond, full_seq=True
            ).reshape(mu.shape[0], -1)
        else:
            alpha = self._weight_model(self.get_y_hist(), cond=cond)
        return alpha

    # --------------- #
    # FILTER/SMOOTHER #
    # --------------- #

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """See parent class."""
        if self._is_continuous:
            mu, cov, Cs = self.vector_to_gaussian_parameters(z, return_Cs=True)

            # checking whether in KL loss function
            alpha = self._kl_loss_detect(mu, cond=cond)
            A = self._dynamics.get_A(alpha)
            A_T = A.transpose(-1, -2)

            dcov = A @ cov + cov @ A_T + self.Q

            cov = 0.5 * (cov + cov.transpose(-1, -2))
            dcov = 0.5 * (dcov + dcov.transpose(-1, -2))

            mu_next = self._dynamics(t, mu, u, self.get_y_hist(), cond=cond)
            c_dot = Cs @ A_T
            dynamics = self.gaussian_parameters_to_vector(
                mu_next, cov, Cs=c_dot, dcov=dcov
            )

        else:
            mu, cov = self.vector_to_gaussian_parameters(z)

            # checking whether in KL loss function
            alpha = self._kl_loss_detect(mu, cond=cond)
            A = self._dynamics.get_A(alpha)
            A_T = A.transpose(-1, -2)

            cov_next = A @ cov @ A_T + self.Q
            cov_next = 0.5 * (cov_next + cov_next.transpose(-1, -2))

            self._smooth_cache.G_tk.append(A)

            mu_next = self._dynamics(t, mu, u, self.get_y_hist(), cond=cond)
            dynamics = self.gaussian_parameters_to_vector(mu_next, cov_next, dcov=None)

        return dynamics

    def measurement_update(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """See parent class."""
        # replace y history with empty tensor to pass batch information to weight model
        if not self.check_y_hist():
            self.set_y_hist(self.y0.repeat(1, y.shape[0], 1))

        # unpack latent belief. Cs is a smoothing parameter.
        if self._is_continuous:
            mu_p, cov_p, Cs = self.vector_to_gaussian_parameters(z, return_Cs=True)
            self._smooth_cache.G_tk.append(Cs @ torch.inverse(cov_p))
            self._smooth_cache.Cs.append(Cs)
        else:
            mu_p, cov_p = self.vector_to_gaussian_parameters(z)

        y_p = self.observation_dynamics(mu_p, cond=cond)  # observe on predicted mean

        # observation Jacobian. NOT the same as Cs above.
        alpha = self._weight_model(self.get_y_hist(), cond=cond)
        C = self._observation_dynamics.get_C(alpha)
        C_T = C.transpose(-1, -2)
        K = cov_p @ C_T @ torch.inverse(C @ cov_p @ C_T + self.R)  # Kalman gain

        # measurement updates. Joseph form for covariance update.
        mu_new = mu_p + (K @ (y - y_p).unsqueeze(-1)).squeeze(-1)
        J = torch.eye(self._latent_dim, device=self._device) - K @ C
        cov_new = J @ cov_p @ J.transpose(-1, -2) + K @ self.R @ K.transpose(-1, -2)
        cov_new = reg_psd(cov_new, reg=self._reg)

        # caching for smoothing
        self._smooth_cache.mu_tk_minus.append(mu_p)
        self._smooth_cache.cov_tk_minus.append(cov_p)
        self._smooth_cache.mu_tk.append(mu_new)
        self._smooth_cache.cov_tk.append(cov_new)

        # update y_hist
        self.set_y_hist(torch.cat([self.get_y_hist(), y.unsqueeze(0)], dim=0))

        return self.gaussian_parameters_to_vector(mu_new, cov_new)


class LinearEnsembleEKFEstimator(EKFEstimator):
    """Estimator for the LE-EKFCell."""

    _cell: LinearEnsembleEKFCell

    def __init__(self, config: "LinearEnsembleEKFEstimatorConfig") -> None:
        """Initialize the LE-EKFEstimator."""
        super(LinearEnsembleEKFEstimator, self).__init__(
            config,
            cell=LinearEnsembleEKFCell(
                num_submodels=config.num_submodels,
                latent_dim=config.latent_dim,
                observation_dim=config.dataset.obs_shape[0],
                cond_dim=config.cond_dim,
                is_continuous=config.is_continuous,
            ),
        )

    @property
    def cell(self) -> LinearEnsembleEKFCell:
        """LE-EKF Cell."""
        return self._cell

    def _update_y_hist(
        self, z_next: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> None:
        """Helper for updating y_hist in predict(...)."""
        # best we can do is observe on the mean, which is passed in from predict()
        y_hist = self.cell.get_y_hist()
        z_mu_next, _ = self.cell.vector_to_gaussian_parameters(z_next)
        yt = self.cell.observation_dynamics(z_mu_next, cond=cond)
        self.cell.set_y_hist(torch.cat([y_hist, yt.unsqueeze(0)], dim=0))

    def predict(
        self,
        z0_mu: torch.Tensor,
        z0_cov: torch.Tensor,
        pred_times: torch.Tensor,
        u: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """See parent class."""
        z0 = self.cell.gaussian_parameters_to_vector(z0_mu, z0_cov)

        # detect kl loss function call
        y_hist = self.cell.get_y_hist()
        if z0.shape[0] > y_hist.shape[1]:
            # history is cat of all observations on every hidden state that was filtered
            p = y_hist.shape[-1]
            reshaped_y_hist = y_hist[:-2].reshape(1, -1, p)  # TODO: fix this 10/24
            self.cell.clear_y_hist(hist_reset_val=reshaped_y_hist)
        else:
            # history is observation on z0 and the observation before z0
            # include 2 because we observe after the state advances in the loop
            # TODO allow more than a history of 2. Currently the full_seq flag requires
            # the state and history time length to be the same
            self.cell.clear_y_hist(hist_reset_val=y_hist[-2:])

        # propagating latent dynamics
        z_next = z0
        z_list = [z_next]

        if self._is_continuous:
            for t_i, t_i_1, u_i in zip(pred_times[:-1], pred_times[1:], u[:-1]):
                # propagate dynamics
                z_next = odeint(
                    lambda t, z: self.cell(t, z, u_i, cond=cond),
                    z_next,
                    torch.tensor(
                        [t_i, t_i_1], dtype=torch.float, device=self.cell._device
                    ),
                    rtol=self._rtol,
                    atol=self._atol,
                )[1, :, :]
                z_list.append(z_next)
                self._update_y_hist(z_next, cond=cond)
        else:
            for t_i, u_i in zip(pred_times[:-1], u[:-1]):
                # propagate dynamics
                z_next = self.cell(t_i, z_next, u_i, cond=cond)
                z_list.append(z_next)
                self._update_y_hist(z_next, cond=cond)

        # stack the latent states
        vectorized_hidden_states = torch.stack(z_list)
        return self._hidden_vector_to_obs(
            vectorized_hidden_states, return_hidden=return_hidden, cond=cond
        )

    def kl_loss(
        self,
        z_mean: torch.Tensor,
        z_cov: torch.Tensor,
        batch_t: torch.Tensor,
        batch_u: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        avg: bool = True,
    ) -> torch.Tensor:
        """Custom KL loss for LE-EKF. See parent function."""
        # preserve the history from before this is called
        old_y_hist = self.cell.get_y_hist()
        loss_dkl = super(LinearEnsembleEKFEstimator, self).kl_loss(
            z_mean, z_cov, batch_t, batch_u, avg=avg, cond=cond
        )
        self.cell.set_y_hist(old_y_hist)
        return loss_dkl


@dataclass(frozen=True)
class LinearEnsembleEKFEstimatorConfig(EKFEstimatorConfig):
    """EKF specific configuration parameters."""

    num_submodels: int = 16

    def create(self) -> LinearEnsembleEKFEstimator:
        """Create LE-EKF from configuration."""
        return LinearEnsembleEKFEstimator(self)
