from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, overload

import torch
from fannypack.nn import resblocks
from fannypack.utils import Buddy
from torch import nn as nn

from dynamics_learning.data.datasets import VisData, VisDataIMG
from dynamics_learning.networks.estimator import Estimator, EstimatorConfig
from dynamics_learning.networks.image_models.kvae import (
    GrayImageVAE,
    _2d_latent_viz,
    _3d_latent_viz,
    _img_seq,
)
from dynamics_learning.utils.log_utils import log_basic, log_image
from dynamics_learning.utils.net_utils import (
    batch_eye,
    gaussian_log_prob,
    reparameterize_gauss,
)


@dataclass(frozen=True)
class LossFeatures:
    """Holds features for computing the loss."""

    q_mu_posterior_list: List[torch.Tensor]
    q_log_var_posterior_list: List[torch.Tensor]
    p_mu_prior_list: List[torch.Tensor]
    p_log_var_prior_list: List[torch.Tensor]

    @staticmethod
    def create_empty() -> "LossFeatures":
        """Create empty LossFeatures."""
        return LossFeatures([], [], [], [])


def make_feature_extractor(in_dim: int, out_dim: int, hidden_units: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_units),
        nn.LeakyReLU(inplace=True),
        resblocks.Linear(hidden_units, activation="leaky_relu"),
        resblocks.Linear(hidden_units, activation="leaky_relu"),
        resblocks.Linear(hidden_units, activation="leaky_relu"),
        nn.Linear(hidden_units, out_dim),
    )


class GSSMBaseline(Estimator):
    """Baseline model.

    Observations are embedded with an MLP.
    These are then passed through an RNN.
    RNN latent states are decoded to produce predicted observations.

    TODO: Document this class better.
    """

    def __init__(self, config: "GSSMBaselineConfig") -> None:
        """Create a GSSMBaseline."""
        super(GSSMBaseline, self).__init__(config)
        self._cond_channels = 0  # for kvae plotting # TODO support conditioning

        self.units = config.hidden_units
        self.obs_latent_dim = config.latent_obs_dim
        self.latent_dim = config.latent_dim
        self._is_image = False
        if len(self.config.dataset.obs_shape) == 1:
            self.obs_dim = self.config.dataset.obs_shape[0]
            # define obs encoder
            self.obs_encoder = make_feature_extractor(
                self.obs_dim, self.obs_latent_dim, self.units
            )
            # define obs decoder
            self.obs_decoder = make_feature_extractor(
                self.latent_dim, self.obs_dim * 2, self.units
            )
        else:
            # Account for images.
            self._image_model = GrayImageVAE(
                self.config.dataset.obs_shape[1],
                self.config.dataset.obs_shape[2],
                self.config.dataset.obs_shape[0],
                self.obs_latent_dim,
                pixel_res=self.config.dataset.pixel_res,  # type: ignore
                use_mmd=False,
                use_dsd=True,
                cond_channels=0,
            )
            # define obs encoder
            self.obs_encoder = nn.Sequential(
                self._image_model._encoder,
                nn.Linear(self.obs_latent_dim * 2, self.obs_latent_dim),
            )
            # define obs decoder
            self.obs_decoder = nn.Sequential(
                nn.Linear(self.latent_dim, self.obs_latent_dim),
                self._image_model._decoder,
            )
            self._is_image = True

        # Create deterministic dynamics
        self.rnn_layers = nn.GRUCell(self.units, hidden_size=self.latent_dim)

        # Create feature extractors
        self.dynamics_distribution = make_feature_extractor(
            self.latent_dim + self.config.ctrl_dim, self.latent_dim * 2, self.units
        )
        self.posterior_encoder = make_feature_extractor(
            self.latent_dim + self.obs_latent_dim, self.latent_dim * 2, self.units
        )

        z0_mu = torch.zeros(self.latent_dim, dtype=torch.float)
        z0_log_var = torch.zeros(self.latent_dim, dtype=torch.float)
        self._z0_mu = torch.nn.Parameter(z0_mu)
        self._z0_log_var = torch.nn.Parameter(z0_log_var)

    def _get_initial_distribution(
        self, batch_size: int, z0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """See parent class."""
        # create initial hidden state
        device = next(self.parameters()).device
        eye = batch_eye(self.latent_dim, batch_size, device=device)
        z0_mu = eye @ self._z0_mu
        z0_log_var = eye @ self._z0_log_var
        return z0_mu, z0_log_var

    def get_initial_hidden_state(
        self, batch_size: int, z0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """See parent class."""
        # create initial hidden state
        device = next(self.parameters()).device
        eye = batch_eye(self.latent_dim, batch_size, device=device)
        z0_mu = eye @ self._z0_mu
        z0_log_var = eye @ self._z0_log_var
        return reparameterize_gauss(z0_mu, z0_log_var, log_flag=True)

    def _dynamics_transition(
        self, t: torch.Tensor, zs_t: torch.Tensor, u_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """See reference.

        https://github.com/google-research/planet/blob/cbe77fc011299becf6c3805d6007c5bf58012f87/planet/models/rssm.py#L96
        """
        # belief, rnn_state are the same
        # https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/rnn/python/ops/gru_ops.py#L197-L214.
        a_t = torch.cat((zs_t, u_t), dim=-1)
        stochastic_distribution = self.dynamics_distribution(a_t)
        mu_zs_t_1 = stochastic_distribution[..., : self.latent_dim]
        log_var_zs_t_1 = stochastic_distribution[..., self.latent_dim :]
        zs_t_1 = reparameterize_gauss(mu_zs_t_1, log_var_zs_t_1, log_flag=True)
        return zs_t_1, mu_zs_t_1, log_var_zs_t_1

    def _measurement_update(
        self, t: torch.Tensor, zd_t_prior: torch.Tensor, y_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """See reference.

        https://github.com/google-research/planet/blob/cbe77fc011299becf6c3805d6007c5bf58012f87/planet/models/rssm.py#L124-L128
        """
        hidden = torch.cat([zd_t_prior, y_t], dim=-1)
        hidden_distribution = self.posterior_encoder(hidden)
        mu_zs = hidden_distribution[..., : self.latent_dim]
        log_var_zs = hidden_distribution[..., self.latent_dim :]
        zs_t_posterior = reparameterize_gauss(mu_zs, log_var_zs, log_flag=True)
        return zs_t_posterior, mu_zs, log_var_zs

    @overload
    def forward(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        u: torch.Tensor,
        z0: torch.Tensor,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """See forward below."""

    @overload
    def forward(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        u: torch.Tensor,
        z0: torch.Tensor,
        return_hidden: bool,
        return_loss_cache: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, LossFeatures]:
        """See forward below."""

    def forward(
        self, t, y, u, z0, return_hidden=False, return_loss_cache=False,
    ):
        """See parent class.

        Note: z_cov is ignored since it has no meaning for this model.

        z0 is of size 2 * latent_dim. The first latent_dim points are the deterministic
        state, the second set of latent_dim points is the stochastic state.
        """
        T, B = y.shape[:2]

        # encode observations
        y_shape = y.shape
        encoded_obs_features = self.obs_encoder(y.reshape(-1, *y_shape[2:])).reshape(
            T, B, -1
        )

        loss_cache = LossFeatures.create_empty()

        mu_zs_0, log_var_zs_0 = self._get_initial_distribution(B)
        loss_cache.p_mu_prior_list.append(mu_zs_0)
        loss_cache.p_log_var_prior_list.append(log_var_zs_0)

        z_next = z0
        zs_list = []
        for t_i, y_i, u_i in zip(t[:-1], encoded_obs_features[:-1], u[:-1]):
            (
                z_next,
                mu_zs_t_posterior,
                log_var_zs_t_posterior,
            ) = self._measurement_update(t_i, z_next, y_i)
            zs_list.append(z_next)

            (
                z_next,
                mu_zs_t_1_prior,
                log_var_zs_t_1_prior,
            ) = self._dynamics_transition(t_i, z_next, u_i)

            # store hidden states and distributions at time i
            loss_cache.q_mu_posterior_list.append(mu_zs_t_posterior)
            loss_cache.q_log_var_posterior_list.append(log_var_zs_t_posterior)
            loss_cache.p_mu_prior_list.append(mu_zs_t_1_prior)
            loss_cache.p_log_var_prior_list.append(log_var_zs_t_1_prior)

        # add trailing measurement update
        zs_next, mu_zs_t_posterior, log_var_zs_t_posterior = self._measurement_update(
            t[-1], z_next, encoded_obs_features[-1]
        )
        zs_list.append(zs_next)
        loss_cache.q_mu_posterior_list.append(mu_zs_t_posterior)
        loss_cache.q_log_var_posterior_list.append(log_var_zs_t_posterior)

        z_all = torch.stack(zs_list)

        # decode predictions
        if return_hidden:
            if return_loss_cache:
                return z_all, torch.eye(self.latent_dim).repeat(T, B, 1, 1), loss_cache
            else:
                return z_all, torch.eye(self.latent_dim).repeat(T, B, 1, 1)
        else:
            if return_loss_cache:
                raise NotImplementedError("This should never be reached.")
            return self.latent_to_observation(z_all, None)

    def predict(
        self,
        z0_mu: torch.Tensor,
        z0_cov: torch.Tensor,
        pred_times: torch.Tensor,
        u: torch.Tensor,
        return_hidden: bool = False,
        cond: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """See parent class.

        Note: z_cov is ignored since it has no meaning for this model.
        """
        B = z0_mu.shape[0]
        zs_list = [z0_mu]

        for t_i, u_i in zip(pred_times[:-1], u[:-1]):
            zs_next, _, _ = self._dynamics_transition(t_i, zs_list[-1], u_i)
            zs_list.append(zs_next)
        z_all = torch.stack(zs_list)

        if return_hidden:
            return (
                z_all,
                torch.eye(self.latent_dim).repeat(len(pred_times), B, 1, 1),
            )
        else:
            return self.latent_to_observation(z_all, None)

    def latent_to_observation(
        self, z_mu: torch.Tensor, z_cov: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """See parent class.

        Note: z_cov is ignored since it has no meaning for this model.
        """
        if self._is_image:
            shape = z_mu.shape
            log_dsd = self.obs_decoder(z_mu.reshape(-1, shape[-1]))
            log_dsd = log_dsd.reshape(*shape[:-1], *log_dsd.shape[-3:])
            return log_dsd
        else:
            decoded_obs_features = self.obs_decoder(z_mu)
            y_mu = decoded_obs_features[..., : self.obs_dim]
            y_cov = torch.diag_embed(
                torch.exp(decoded_obs_features[..., self.obs_dim :])
            )
            return y_mu, y_cov

    def loss(
        self,
        batch_t: torch.Tensor,
        batch_y: torch.Tensor,
        batch_u: torch.Tensor,
        iteration: int,
        avg: bool = True,
    ) -> torch.Tensor:
        """See parent class."""
        T, B = batch_y.shape[:2]
        z0_p = self.get_initial_hidden_state(B)
        loss_cache: LossFeatures
        z_mean, z_cov, loss_cache = self(
            batch_t, batch_y, batch_u, z0_p, return_hidden=True, return_loss_cache=True
        )

        if self._is_image:
            log_dsd = self.latent_to_observation(z_mean, z_cov)
            reconstruction_loss = self._image_model.get_reconstruction_loss(
                batch_y, log_dsd, avg=False
            ).reshape(T, B)

        else:
            y_mean, y_cov = self.latent_to_observation(z_mean, z_cov)
            reconstruction_loss = -gaussian_log_prob(y_mean, y_cov, batch_y)

        z_mean = torch.stack(loss_cache.q_mu_posterior_list)
        z_log_var = torch.stack(loss_cache.q_log_var_posterior_list)
        z_mean_prior = torch.stack(loss_cache.p_mu_prior_list)
        z_log_var_prior = torch.stack(loss_cache.p_log_var_prior_list)

        z_posterior = torch.distributions.normal.Normal(
            z_mean, torch.exp(z_log_var) ** 0.5 + 1e-6
        )
        z_prior = torch.distributions.normal.Normal(
            z_mean_prior, torch.exp(z_log_var_prior) ** 0.5 + 1e-6
        )
        kl = torch.distributions.kl.kl_divergence(z_posterior, z_prior).mean(-1)
        # Reference for kl divergence computation
        # https://github.com/google-research/planet/blob/cbe77fc011299becf6c3805d6007c5bf58012f87/planet/models/rssm.py#L87-L94

        if avg:
            return torch.sum(reconstruction_loss + kl) / (T * B)
        else:
            return reconstruction_loss + kl

    def log(self, buddy: Buddy, viz: VisData, filter_length: int = 1) -> None:
        """Log information and visualization information.

        Parameters
        ----------
        See parent class.
        """
        if self._is_image:
            assert isinstance(viz, VisDataIMG)

            predict_length = min(2 * filter_length, viz.t.shape[0] - filter_length)
            total_len = filter_length + predict_length
            p_img, _ = _img_seq(self, viz, filter_length, predict_length)
            log_image(
                buddy, p_img, "image_trajectories", scope="sequence_visualizations"
            )
            # ---latent visualization--- #
            # smoothed samples
            z0 = self.get_initial_hidden_state(viz.y.shape[1])
            z_mu, _ = self(
                viz.t[:total_len],
                viz.y[:total_len],
                viz.u[:total_len],
                z0,
                return_hidden=True,
            )
            z_samp_s = z_mu.cpu().numpy()

            if self.latent_dim * 2 == 2:
                p_img_s, _ = _2d_latent_viz(z_samp_s, "Filtered Latent Trajectories")
                ftitle = "filtered_latent_trajectories"
                log_image(buddy, p_img_s, f"{ftitle}", scope="latent_viz")

            else:
                # take the first 3
                p_img_s, _ = _3d_latent_viz(
                    z_samp_s[:3], "Filtered Latent Trajectories"
                )
                ftitle = "filtered_latent_trajectories"
                log_image(buddy, p_img_s, f"{ftitle}", scope="latent_viz")
        else:
            log_basic(self, buddy, viz, filter_length=filter_length)
            viz_loss = self.loss(viz.t, viz.y, viz.u, 0)  # iteration doesn't matter
            with buddy.log_scope("Validation_BASELINE"):
                buddy.log_scalar("Validation_Loss", viz_loss.item())

    def eval_loss(self, viz: VisData, filt_points: int, pred_points: int) -> None:
        """Prints evaluation losses for the model.

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

        # pend_img
        if isinstance(viz, VisDataIMG):
            # filtering and prediction time/data
            t_filt = viz.t[:filt_points]
            o_filt = viz.y[:filt_points]
            u_filt = viz.u[:filt_points]
            t_pred = viz.t[(filt_points - 1) : (filt_points + pred_points - 1)]
            o_pred = viz.y[(filt_points - 1) : (filt_points + pred_points - 1)]
            u_pred = viz.u[(filt_points - 1) : (filt_points + pred_points - 1)]

            T, B = o_pred.shape[:2]

            # filtering
            T, B = viz.y.shape[:2]
            z0_f = self.get_initial_hidden_state(B)
            z_mu_f, z_cov_f = self(t_filt, o_filt, u_filt, z0_f, return_hidden=True)

            # prediction
            z0_mu_p = z_mu_f[-1]
            z0_cov_p = z_cov_f[-1]

            # prediction
            if False:  # TODO handle conditioning
                _o_pred = o_pred[:, :, : -self._cond_channels, 0, 0]
                cond = o_pred[0, :, -self._cond_channels :, 0, 0]  # conditional context
            else:
                _o_pred = o_pred
                cond = None

            # computing l2 loss over 100 samples
            log_dsd_p = self.predict(z0_mu_p, z0_cov_p, t_pred, u_pred, cond=cond)
            _img_samples = []
            for i in range(100):
                if i % 10 == 0:
                    print(f"Sample {i} / 100")
                _img_samples.append(self._image_model.sample_img(log_dsd_p))
            img_samples = torch.stack(_img_samples)
            loss_l2 = torch.mean(torch.sqrt((img_samples - _o_pred.unsqueeze(0)) ** 2))

            # NLL Loss on discrete log-softmax distribution
            o_quant = (_o_pred.squeeze(2) * (self._image_model.pixel_res - 1)).long()
            o_quant = o_quant.reshape(-1, *_o_pred.shape[-2:])
            logits = log_dsd_p.reshape(
                -1, *log_dsd_p.shape[-3:]
            )  # (B, num_cats, img_h, img_w)

            # average loss over batches. dkl already batch-averaged.
            nll_loss_func = nn.NLLLoss(reduction="sum")
            nll = nll_loss_func(logits, o_quant) / logits.shape[0]

            # # reporting the evaluation
            print(
                f"Prediction Loss (filt_pts={filt_points}, pred_pts={pred_points}) \t"
                f"L2 Loss: {loss_l2.item():.3f} \t"
                f"NLL: {nll.item():.3f}"
            )

        # stripped datasets
        else:
            # filtering and prediction time/data
            t_filt = viz.t[:filt_points]
            y_filt = viz.y[:filt_points]
            u_filt = viz.u[:filt_points]
            t_pred = viz.t[(filt_points - 1) : (filt_points + pred_points - 1)]
            y_pred = viz.y[(filt_points - 1) : (filt_points + pred_points - 1)]
            u_pred = viz.u[(filt_points - 1) : (filt_points + pred_points - 1)]

            # filtering
            B = viz.y.shape[1]
            z0_f = self.get_initial_hidden_state(B)
            z_mu_f, z_cov_f = self(t_filt, y_filt, u_filt, z0_f, return_hidden=True)

            # prediction
            z0_mu_p = z_mu_f[-1]
            z0_cov_p = z_cov_f[-1]

            y_samples = []
            for i in range(100):
                if i % 10 == 0:
                    print(f"Rollout {i} / 100")
                y_mu_p, y_cov_p = self.predict(z0_mu_p, z0_cov_p, t_pred, u_pred)
                y_sample = reparameterize_gauss(y_mu_p, y_cov_p)
                y_samples.append(y_sample)

            y_samples_torch = torch.stack(y_samples)
            mean = y_samples_torch.mean(dim=0)
            var = y_samples_torch.var(dim=0)

            # computing losses (NLL and L2)
            loss_nll = -torch.mean(
                gaussian_log_prob(mean, torch.diag_embed(var), y_pred)
            )
            loss_ade = torch.mean(
                torch.sqrt((y_samples_torch - y_pred.unsqueeze(0)) ** 2)
            )

            # reporting the evaluation
            print(
                f"Prediction Loss (filt_pts={filt_points}, pred_pts={pred_points}) \t"
                f"NLL Loss: {loss_nll.item():.3f} \t ADE Loss: {loss_ade.item():.5f}"
            )

    def summary_plot(self, viz: VisData, name: str, debug: bool = False) -> None:
        """Produces and saves summary plots. Will produce same plots as log().

        See parent class.
        """
        # pend_img
        if isinstance(viz, VisDataIMG):
            DIR_NAME = "summary_plots/" + name
            Path(DIR_NAME).mkdir(parents=True, exist_ok=True)
            filter_length = 25
            pred_length = 50

            # ---image sequences--- #
            _, seq_fig = _img_seq(self, viz, filter_length, pred_length)
            seq_fig.savefig(DIR_NAME + "/img_seq")
        else:
            super(GSSMBaseline, self).summary_plot(viz, name, debug)


@dataclass(frozen=True)
class GSSMBaselineConfig(EstimatorConfig):
    """Baseline specific configuration parameters."""

    latent_obs_dim: int
    hidden_units: int

    def create(self) -> GSSMBaseline:
        """See parent."""
        return GSSMBaseline(self)
