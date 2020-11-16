from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib
import numpy as np
import torch
from fannypack.utils import Buddy, freeze_module, to_numpy
from matplotlib import axes, gridspec
from matplotlib import lines as mlines
from matplotlib import patches
from matplotlib import pyplot as plt
from torch import nn as nn
from torch.nn import functional as F

from dynamics_learning.data.datasets import (
    ImageDynamicDatasetConfig,
    VisData,
    VisDataIMG,
)
from dynamics_learning.networks.estimator import Estimator, EstimatorConfig
from dynamics_learning.networks.image_models.conv_networks import (
    get_simple_encoder_and_dsd_decoder,
)
from dynamics_learning.networks.image_models.external_models import (
    FullyConvDecoderVAE,
    FullyConvEncoderVAE,
)
from dynamics_learning.networks.kalman.core import (
    KalmanEstimator,
    KalmanEstimatorConfig,
)
from dynamics_learning.networks.models import MLP
from dynamics_learning.utils.log_utils import log_image, log_scalars
from dynamics_learning.utils.net_utils import (
    dkl_diag_gaussian,
    gaussian_log_prob,
    reparameterize_gauss,
)
from dynamics_learning.utils.plot_utils import PlotHandler as ph


@dataclass
class KVAEDistributions:
    """Container for passing KVAE distributions.

    Holds the following:

    USED FOR LOSS:
    - image recognition model           : q(y | o), diagonal-cov Gaussian
    - image generative model            : p(o | y), discrete softmax distribution
    - smoothed latent distribution      : q(z | y), Gaussian
    - smoothed observation distribution : p(y | z), Gaussian

    USED FOR LOGGING/OTHER ANALYSIS:
    - filtered latent distribution      : q(z | y), Gaussian
    - filtered observation distribution : p(y | z), Gaussian
    - filtered image generative model   : p(o | y), discrete softmax distribution
    - smoothed image generative model   : p(o | y), discrete softmax distribution

    Notation
    --------
    y_mu_r : torch.Tensor, shape=(..., p)
        Mean of q(y | o).
    y_log_sigma_r : torch.Tensor, shape=(..., p)
        Log variances of q(y | o).
    log_dsd_r : torch.Tensor, shape=(..., pixel_res, image_height, image_width)
        Log of p(o | y), a discrete softmax distribution. Formed by directly
        reconstructing the input data (only a reflection of the VAE).
    z_mu_f : torch.Tensor, shape=(..., n)
        Mean of p(z | y) from filter.
    z_cov_f : torch.Tensor, shape=(..., n, n)
        Covariance of p(z | y) from filter.
    y_mu_f : torch.Tensor, shape=(..., p)
        Mean of p(y | z) from filter.
    y_cov_f : torch.Tensor, shape=(..., p, p)
        Covariance of p(y | z) from filter.
    log_dsd_f : torch.Tensor, shape=(..., pixel_res, image_height, image_width)
        Log of p(o | y), a discrete softmax distribution. Formed by decoding samples
        from the filtering distributions.
    z_mu_s : torch.Tensor, shape=(..., n)
        Mean of p(z | y) from smoother.
    z_cov_s : torch.Tensor, shape=(..., n, n)
        Covariance of p(z | y) from smoother.
    y_mu_s : torch.Tensor, shape=(..., p)
        Mean of p(y | z) from smoother.
    y_cov_s : torch.Tensor, shape=(..., p, p)
        Covariance of p(y | z) from smoother.
    log_dsd_s : torch.Tensor, shape=(..., pixel_res, image_height, image_width)
        Log of p(o | y), a discrete softmax distribution. Formed by decoding samples
        from the smoothing distributions.
    """

    y_mu_r: torch.Tensor  # q(y | o)
    y_log_sigma_r: torch.Tensor
    log_dsd_r: torch.Tensor  # p(o | y)
    z_mu_f: torch.Tensor  # q(z | y)
    z_cov_f: torch.Tensor
    y_mu_f: torch.Tensor  # p(y | z)
    y_cov_f: torch.Tensor
    log_dsd_f: torch.Tensor
    z_mu_s: torch.Tensor  # q(z | y)
    z_cov_s: torch.Tensor
    y_mu_s: torch.Tensor  # p(y | z)
    y_cov_s: torch.Tensor
    log_dsd_s: torch.Tensor


class GrayImageVAE(nn.Module):
    """VAE for 2D grayscale (single-channel) image recognition.

    TODO: probably move this into its own file and also rename it to ImageVAE.
    """

    def __init__(
        self,
        img_height: int,
        img_width: int,
        img_channels: int = 1,
        latent_dim: int = 2,
        pixel_res: int = 256,
        use_mmd: bool = False,
        use_dsd: bool = False,
        cond_channels: int = 0,
    ) -> None:
        """Initialize the gray image VAE.

        Expects to take in data of shape (..., 1, img_height, img_width).
        The 1 channel represents the single color channel.

        Parameters
        ----------
        img_height : int
            Height of the image (number of rows).
        img_width : int
            Width of the image (number of columns).
        img_channels : int
            Number of channels  of the image.
        latent_dim : int, default=None
            Latent dimension. If None, default (2) is used.
        pixel_res : int, default=None
            Pixel resolution. If None, default (256) is used.
        use_mmd : bool, default=False
            Use mmd loss as in InfoVAE.
        use_dsd : bool, default=False
            Use dsd in the output distribution of the VAE.
        cond_channels : int, default=0
            Extra conditioning channels. We assume that if there are conditioning
            channels, then the data (images) are already augmented with conditioning
            channels and we can read off the one-hot vectors from the data.
        """
        super(GrayImageVAE, self).__init__()

        # attributes
        self.pixel_res = pixel_res
        self._img_height = img_height
        self._img_width = img_width
        self._img_channels = img_channels
        self._latent_dim = latent_dim
        self._pixel_res = pixel_res
        self._use_mmd = use_mmd
        self._use_dsd = use_dsd
        self._cond_channels = cond_channels

        if self._use_dsd:
            # use the discrete softmax distribution over pixels
            assert self._img_height == self._img_width
            channels = [32 * (2 ** i) for i in range(int(np.log2(self._img_height)))]
            self._encoder, self._decoder = get_simple_encoder_and_dsd_decoder(
                img_channels,
                channels,
                self._img_height,
                latent_dim,
                pixel_res,
                cond_channels=cond_channels,
            )
        else:
            # use BCE loss and external VAE architecture
            self._encoder = FullyConvEncoderVAE(
                input=img_channels + cond_channels,  # conv layer first
                latent_size=self._latent_dim,
                bn=True,
                drop=False,
                nl=nn.ReLU(),
                img_dim=str(self._img_height),
                extra_scalars=0,
                extra_scalars_conc=0,
                stochastic=True,
            )
            self._decoder = FullyConvDecoderVAE(
                input=img_channels,
                latent_size=self._latent_dim + cond_channels,  # linear layer first
                bn=True,
                img_dim=str(self._img_height),
                drop=False,
                nl=nn.ReLU(),
                output_nl=None,
            )

        # initialize prior network if you have conditioning channels
        # takes in only a one-hot vector of length num_cats and gives a latent dist
        if cond_channels > 0:
            self._prior = MLP(cond_channels, 2 * self._latent_dim, [64] * 3, nn.ReLU())
        else:
            self._prior = None

    @property
    def _device(self):
        """Return the device."""
        return next(self.parameters()).device

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """Encode an image.

        Parameters
        ----------
        img : torch.Tensor, shape=(B, C, img_height, img_width)
            Input image.

        Returns
        -------
        out : torch.Tensor, shape=(B, 2 * n)
            Flattened latent variable Gaussian distribution parameters.
        """
        return self._encoder(img)

    def decode(
        self, z: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode a latent variable.

        Parameters
        ----------
        z : torch.Tensor, shape=(B, n)
            Latent variable.
        cond : Optional[torch.Tensor], shape=(B, num_cats)
            Conditioning one-hot vectors.

        Returns
        -------
        log_dsd : torch.Tensor, shape=(B, pixel_res, img_height, img_width)
            Flattened discrete log-softmax distribution over pixels. In the form
            usable by nn.NLLLoss(). The "classes" are grayscale pixel values in
            the -3 dimension of log_dsd.
        """
        if cond is None:
            _input = z
        else:
            assert self._prior is not None
            _input = torch.cat([z, cond], dim=-1)
        return self._decoder(_input)

    def sample_latent(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples latent variables given data.

        Parameters
        ----------
        img : torch.Tensor, shape=(..., C, img_height, img_width)
            Input image. C could also include conditional channels.

        Returns
        -------
        z : torch.Tensor, shape=(..., n)
            Sampled latent variables.
        mu_e : torch.Tensor, shape=(..., n), OPTIONAL
            Encoder mean.
        log_var_e : torch.Tensor, shape=(..., n), OPTIONAL
            Flattened encoder log-variances of each latent variable.
        """
        img_shape = img.shape[-3:]
        leading_shape = img.shape[:-3]
        img = img.reshape(-1, *img_shape)  # (B, C, H, W)
        out_enc = self.encode(img).reshape(*leading_shape, -1)
        mu_e = out_enc[..., : self._latent_dim]
        log_var_e = out_enc[..., self._latent_dim :]
        z = reparameterize_gauss(mu_e, log_var_e, log_flag=True, device=self._device)
        return z, mu_e, log_var_e

    def sample_img(self, log_dsd: torch.Tensor) -> torch.Tensor:
        """Samples an image from a discrete log-softmax distribution.

        Parameters
        ----------
        log_dsd : torch.Tensor, shape=(..., pixel_res, img_height, img_width)
            Discrete log softmax distribution over pixels.

        Returns
        -------
        img : torch.Tensor, shape=(..., C, img_height, img_width)
            Sampled image.
        """
        if self._use_dsd:
            # Gumbel-softmax sampling (but pass out hard vector, + ensures quantization)
            one_hot = F.gumbel_softmax(
                log_dsd, tau=0.1, hard=True, dim=-3
            )  # one-hot vector
            pixel_vals = torch.linspace(0.0, 1.0, self.pixel_res, device=self._device)
            # computes the pixel value by dotting one_hot with pixel_vals
            # unsqueeze(-3) to add back the reduced single color channel
            img = torch.tensordot(pixel_vals, one_hot, dims=([0], [-3])).unsqueeze(-3)
        else:
            # # DEBUG: BCE loss and continuous values
            img = torch.round(torch.sigmoid(log_dsd) * (self._pixel_res - 1)) / (
                self._pixel_res - 1
            )
        return img

    def forward(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """VAE forward pass.

        Parameters
        ----------
        img : torch.Tensor, shape=(..., C, img_height, img_width)
            Input image.

        Returns
        -------
        mu_e : torch.Tensor, shape=(..., n)
            Encoder mean.
        log_var_e : torch.Tensor, shape=(..., n)
            Flattened encoder log-variances of each latent variable.
        log_dsd : torch.Tensor, shape=(..., pixel_res, img_height, img_width)
            Discrete log softmax distribution over pixels.
        """
        z, mu_e, log_var_e = self.sample_latent(img)  # automatically handles cond

        # check whether VAE is conditional
        cond = None if self._prior is None else img[..., -self._cond_channels :, 0, 0]
        log_dsd = self.decode(z, cond=cond)

        return mu_e, log_var_e, log_dsd

    def generate(
        self,
        B: int = 1,
        z: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        device: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates image samples.

        Parameters
        ----------
        B : int, default = 1
            Size of batch to be generated.
        z : Optional[torch.Tensor], shape=(B, n), default=None
            Latent variable. If None, one is sampled from a standard normal prior.
        cond : Optional[torch.Tensor], shape=(B, cond_channels), default=None
            Conditioning vector.
        device : Optional[str], default=None
            Name of device.

        Returns
        -------
        img : torch.Tensor, shape=(B, C, img_height, img_width)
            Batch of generated images.
        log_dsd : torch.Tensor, shape=(B, pixel_res, img_height, img_width), OPTIONAL
            Discrete log-softmax distribution over pixels.
        """
        if z is None:
            assert device is not None
            eps = torch.randn(B, self._latent_dim, device=device)  # noise

            if self._prior is not None:
                assert cond is not None
                out_p = self._prior(cond)
                mu_p = out_p[..., : self._latent_dim]
                log_var_p = out_p[..., self._latent_dim :]
                _z = mu_p + torch.exp(log_var_p) * eps
            else:
                _z = eps
        else:
            assert z.shape == (B, self._latent_dim)
            _z = z

        if cond is not None:
            assert cond.shape[0] == B

        log_dsd = self.decode(_z, cond=cond)
        img = self.sample_img(log_dsd)
        return img, log_dsd

    def reconstruct(self, img: torch.Tensor) -> torch.Tensor:
        """Reconstruct image sequence.

        Parameters
        ----------
        img : torch.Tensor, shape=(B, C, img_height, img_width)
            Input images.
        """
        B = img.shape[0]
        z, _, _ = self.sample_latent(img)
        cond = None if self._prior is None else img[..., -self._cond_channels :, 0, 0]
        img_recon, _ = self.generate(B=B, z=z, cond=cond)
        return img_recon

    # ------------------------- #
    # LOSS FUNCTION AND HELPERS #
    # ------------------------- #

    def _compute_kernel(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes Gaussian kernel between two tensors. Used for MMD loss.

        Source: https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb

        Parameters
        ----------
        z1 : torch.Tensor, shape=(B, n)
            First sample.
        z2 : torch.Tensor, shape=(B, n)
            Second sample.

        Returns
        -------
        k : torch.Tensor, shape=(1)
            Kernel function value.
        """
        assert z1.shape == z2.shape
        B, n = z1.shape

        z1_tiled = z1.unsqueeze(1).expand(B, B, n)
        z2_tiled = z2.unsqueeze(0).expand(B, B, n)
        kernel_input = (z1_tiled - z2_tiled).pow(2).mean(2) / float(n)
        return torch.exp(-kernel_input)

    def _compute_mmd(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes the MMD loss.

        Source: https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb

        Parameters
        ----------
        z1 : torch.Tensor, shape=(..., n)
            First sample.
        z2 : torch.Tensor, shape=(..., n)
            Second sample.

        Returns
        -------
        loss_mmd : torch.Tensor, shape=(1)
            MMD loss.
        """
        z1_r = z1.reshape(-1, z1.shape[-1])
        z2_r = z2.reshape(-1, z2.shape[-1])
        z1_kernel = self._compute_kernel(z1_r, z1_r)
        z2_kernel = self._compute_kernel(z2_r, z2_r)
        z1z2_kernel = self._compute_kernel(z1_r, z2_r)
        loss_mmd = z1_kernel.mean() + z2_kernel.mean() - 2.0 * z1z2_kernel.mean()
        return loss_mmd

    def get_reconstruction_loss(
        self, _img: torch.Tensor, log_dsd: torch.Tensor, avg: bool
    ) -> torch.Tensor:
        """Image reconstruction loss."""
        # reconstruction loss term
        if self._use_dsd:
            # convert img to categorical data
            # assumes pixel values are already quantized to the proper resolution!
            categorical_img = (_img * (self.pixel_res - 1)).long()  # (B, img_h, img_w)
            categorical_img = categorical_img.reshape(-1, *_img.shape[-2:])
            logits = log_dsd.reshape(
                -1, *log_dsd.shape[-3:]
            )  # (B, num_cats, img_h, img_w)

            # average loss over batches. dkl already batch-averaged.
            nll_loss_func = nn.NLLLoss(reduction="none")
            loss_recon = torch.sum(nll_loss_func(logits, categorical_img), dim=(-2, -1))
            if avg:
                loss_recon = torch.sum(loss_recon) / logits.shape[0]
        else:
            # Architecture test, log_dsd is unnormalized logits
            loss_REC = nn.BCEWithLogitsLoss(reduction="sum")
            loss_recon = loss_REC(log_dsd, _img) / _img.shape[0]

        return loss_recon

    def loss(
        self,
        img: torch.Tensor,
        mu_e: torch.Tensor,
        log_var_e: torch.Tensor,
        log_dsd: torch.Tensor,
        avg: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the VAE loss for image reconstruction.

        Compatible with inputs at single points in time or sequences
        of many times. Will collapse preceding dimensions into a
        single batch dimension.

        Parameters
        ----------
        img : torch.Tensor, shape=(..., C, img_height, img_width)
            Input image.
        mu_e : torch.Tensor, shape=(..., n)
            Encoder mean.
        log_var_e : torch.Tensor, shape=(..., n)
            Encoder log-variances.
        log_dsd : torch.Tensor, shape=(..., pixel_res, img_height, img_width)
            Discrete log-softmax distribution over pixel values (decoder).
        avg : bool, default=True
            Flag indicating whether to average loss components by batch and time.

        Returns
        -------
        loss_recon : torch.Tensor, shape=(1)
            Reconstruction loss averaged over batches.
        loss_reg : torch.Tensor, shape=(1)
            Regularizing loss term (either KL or MMD).
        """
        n = mu_e.shape[-1]

        # checking whether VAE is conditional
        if self._prior is not None:
            _img = img[..., : -self._cond_channels, :, :]
            cond = img[..., -self._cond_channels :, 0, 0]
        else:
            _img = img[..., :, :, :]

        loss_recon = self.get_reconstruction_loss(_img, log_dsd, avg=avg)

        # regularization loss term
        loss_reg: torch.Tensor
        if self._use_mmd:
            z_model, _, _ = self.sample_latent(img)
            eps = torch.randn_like(z_model, requires_grad=False, device=self._device)

            if self._prior is not None:
                out_p = self._prior(cond)
                mu_p = out_p[..., : self._latent_dim]
                log_var_p = out_p[..., self._latent_dim :]
                z_prior = mu_p + torch.exp(log_var_p) * eps
            else:
                z_prior = eps
            loss_reg = self._compute_mmd(z_prior, z_model)
        else:
            if self._prior is not None:
                out_p = self._prior(cond)
                mu_p = out_p[..., : self._latent_dim]
                log_var_p = out_p[..., self._latent_dim :]
                loss_reg = dkl_diag_gaussian(
                    mu_e.reshape(-1, n),
                    log_var_e.reshape(-1, n),
                    mu_2=mu_p.reshape(-1, n),
                    var_2=log_var_p.reshape(-1, n),
                    log_flag=True,
                )
            else:
                loss_reg = dkl_diag_gaussian(
                    mu_e.reshape(-1, n), log_var_e.reshape(-1, n), log_flag=True,
                )

        return loss_recon, loss_reg


class KVAE(Estimator):
    """Kalman VAE.

    References
    ----------
    [1] "A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning"
    """

    def __init__(
        self,
        config: "KVAEConfig",
        img_height: int,
        img_width: int,
        img_channels: int,
        pixel_res: int,
        use_dsd: bool,
        kf: KalmanEstimator,
        image_model: Optional[nn.Module] = None,
        image_model_path: Optional[Tuple[str, bool]] = None,
        cond_channels: int = 0,
    ) -> None:
        """Initialize the KVAE.

        Parameters
        ----------
        img_height : int
            Height of the image.
        img_width : int
            Width of the image.
        img_channels : int
            Number of channels  of the image.
        pixel_res : int
            Pixel resolution. This is the number of discrete bins for a given pixel.
        use_dsd : bool
            Use dsd in the output distribution of the VAE.
        kf : KalmanEstimator
            The latent filter model.
        image_model : nn.Module, default=None
            The image VAE. If None, a default will be generated.
        image_model_path : Tuple[str, bool], default=NONE
            Path containing pretrained image network parameters and a bool indicating if the model should be frozen or
            not. True means the model is frozen.
        cond_channels : int, default=0
            Number of conditional channels for context.
        """
        super(KVAE, self).__init__(config)

        # attributes
        self._img_height = img_height
        self._img_width = img_width
        self._img_channels = img_channels
        self._cond_channels = cond_channels

        # models
        assert kf._is_smooth  # enforce Kalman smoothing
        self._kf = kf
        if image_model is None:
            self._image_model = GrayImageVAE(
                img_width,
                img_height,
                img_channels,
                kf.cell._observation_dim,
                pixel_res=pixel_res,
                use_mmd=False,
                use_dsd=use_dsd,
                cond_channels=cond_channels,
            )
            if image_model_path is not None:
                self._image_model.load_state_dict(torch.load(image_model_path[0]))
                if image_model_path[1]:
                    freeze_module(self._image_model)
        else:
            self._image_model = image_model

    @property
    def _device(self):
        """The device."""
        return next(self.parameters()).device

    def forward(
        self,
        times: torch.Tensor,
        img: torch.Tensor,
        u: torch.Tensor,
        z0: Optional[torch.Tensor] = None,
    ) -> Tuple[KVAEDistributions, torch.Tensor]:
        """Forward pass of the KVAE.

        Computes distributions to pass to the loss function. See the docstring for the
        KVAEDistributions dataclass.

        Parameters
        ----------
        times : torch.Tensor, shape=(T)
            List of times.
        img : torch.Tensor, shape=(T, B, C, img_height, img_width)
            Batch of image trajectories.
        u : torch.Tensor, shape=(T, B, m)
            Control inputs.
        z0 : Optional[torch.Tensor], shape=(B, n), default=None
            Initial latent state for KF. If None, one is generated with the KF.

        Returns
        -------
        pack : KVAEDistributions
            Packed container with all the required distributions.
        y_r : torch.Tensor, shape=(T, B, p)
            Sample of image embedding from recognition net, y_r ~ q(y | o)
        """
        # shapes
        T, B = img.shape[:2]
        p = self._image_model._latent_dim

        # checking for conditional context
        if self._cond_channels > 0:
            cond_all = img[..., -self._cond_channels :, 0, 0]
            cond = cond_all[0]  # (B, cond_dim)
            cond_all_flat = cond_all.reshape(
                -1, self._cond_channels
            )  # (T * B, cond_dim)
        else:
            cond_all_flat = None
            cond_all = None
            cond = None

        # image recognition network, _r denotes recognition, q(y | o)
        y_r, y_mu_r, y_log_sigma_r = self._image_model.sample_latent(img)

        # image reconstruction distribution
        log_dsd_r = self._image_model.decode(y_r.reshape(-1, p), cond=cond_all_flat)
        log_dsd_r = log_dsd_r.reshape(T, B, *log_dsd_r.shape[-3:])  # p(o | y)

        # filter/smoother, _f denotes filter, _s denotes smoother. in order:
        # p(z), q(z | y), p(y | z)
        if z0 is None:
            _z0 = self._kf.cell.get_initial_hidden_state(B)
        else:
            _z0 = self._kf.cell.get_initial_hidden_state(B, z0=z0)

        # must run filter pre-smoothing
        z_mu_f, z_cov_f = self._kf(times, y_r, u, _z0, cond=cond, return_hidden=True)
        y_mu_f, y_cov_f = self._kf.latent_to_observation(z_mu_f, z_cov_f, cond=cond_all)
        z_mu_s, z_cov_s = self._kf.get_smooth()
        y_mu_s, y_cov_s = self._kf.latent_to_observation(z_mu_s, z_cov_s, cond=cond_all)

        # image distributions from filter and smoother
        y_f = reparameterize_gauss(y_mu_f, y_cov_f)
        y_s = reparameterize_gauss(y_mu_s, y_cov_s)

        log_dsd_f = self._image_model.decode(y_f.reshape(-1, p), cond=cond_all_flat)
        log_dsd_f = log_dsd_f.reshape(T, B, *log_dsd_f.shape[-3:])
        log_dsd_s = self._image_model.decode(y_s.reshape(-1, p), cond=cond_all_flat)
        log_dsd_s = log_dsd_s.reshape(T, B, *log_dsd_s.shape[-3:])

        # pack to cleanly pass to loss
        pack = KVAEDistributions(
            y_mu_r,
            y_log_sigma_r,
            log_dsd_r,
            z_mu_f,
            z_cov_f,
            y_mu_f,
            y_cov_f,
            log_dsd_f,
            z_mu_s,
            z_cov_s,
            y_mu_s,
            y_cov_s,
            log_dsd_s,
        )

        return pack, y_r

    def predict(
        self,
        z0_mu: torch.Tensor,
        z0_cov: torch.Tensor,
        pred_times: torch.Tensor,
        u: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict a sequence of images given an initial latent condition.

        Parameters
        ----------
        See parent method.

        cond : Optional[torch.Tensor], shape=(B, cond_dim), default=None
            Conditional context.

        Returns
        -------
        log_dsd, torch.Tensor, shape=(T, B, pixel_res, img_height, img_width)
            Batch of sequences of discrete log-softmax distributions over reconstructions.
        """
        # checking for conditional context
        if self._cond_channels > 0:
            assert cond is not None
            cond_all = cond.unsqueeze(0).repeat(len(pred_times), 1, 1)
        else:
            cond_all = None

        z_mu, z_cov = self._kf.predict(
            z0_mu, z0_cov, pred_times, u, cond=cond, return_hidden=True
        )
        return self.latent_to_observation(z_mu, z_cov, cond=cond_all)

    def get_initial_hidden_state(
        self, batch_size: int, z0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Always returns None. Here for API compliance, z0 generated internally."""
        return None

    def latent_to_observation(
        self,
        z_mu: torch.Tensor,
        z_cov: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Converts a sequence of latent distributions into observation distributions.

        The distribution over pixels will be stochastically estimated using a
        one-sample estimate of the intermediate latent observation y.

        Parameters
        ----------
        z_mu : torch.Tensor, shape=(T, B, n)
            Latent means.
        z_cov : torch.Tensor, shape=(T, B, n, n)
            Latent covariances.
        cond : Optional[torch.Tensor], shape=(T, B, cond_dim)
            Conditional context.

        Returns
        -------
        log_dsd : torch.Tensor, shape=(T, B, pixel_res, img_height, img_width)
            Batch of sequences of discrete log-softmax distributions over reconstructions.
        """
        # shapes
        T, B = z_mu.shape[:2]
        p = self._image_model._latent_dim  # latent observation dim

        # image model needs flattened batches, so T and B combined
        y_mu, y_cov = self._kf.latent_to_observation(z_mu, z_cov, cond=cond)
        y_cov = 0.5 * (y_cov + y_cov.transpose(-1, -2))
        y_sample = reparameterize_gauss(y_mu, y_cov, device=self._device)
        if cond is not None:
            cond_reshape = cond.reshape(T * B, -1)
        else:
            cond_reshape = None
        log_dsd = self._image_model.decode(y_sample.reshape(-1, p), cond_reshape)
        log_dsd = log_dsd.reshape(T, B, *log_dsd.shape[-3:])
        return log_dsd

    def loss(
        self,
        batch_t: torch.Tensor,
        batch_o: torch.Tensor,
        batch_u: torch.Tensor,
        iteration: int,
        avg: bool = True,
        return_components: bool = False,
    ) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """See parent class.

        See [1] sec. 3.2 eqn. (6). Notation in this code differs from paper.

        Paper -> Code
        -------------
        x -> y
        a -> o
        z -> z

        For more notation details see docstring for forward().

        Parameters
        ----------
        batch_t : torch.Tensor, shape=(T)
            Evaluation times.
        batch_o : torch.Tensor, shape=(..., C, img_height, img_width)
            Collection of gray image data.
        batch_u : torch.Tensor, shape=(..., m)
            Batch of control inputs.
        iteration : int
            Current training iteration.
        avg : bool, default=True
            Flag indicating whether to take the average of the loss.
        return_components : bool, default=False
            Flag indicating whether to return loss components.

        Returns
        -------
        loss : torch.Tensor, shape=(1)
            KVAE loss averaged over batches and time.
        """
        T, B = batch_o.shape[:2]
        pack, y_r = self(batch_t, batch_o, batch_u)

        # log p(o | y) - image model
        loss_r_img, _ = self._image_model.loss(
            batch_o, pack.y_mu_r, pack.y_log_sigma_r, pack.log_dsd_r, avg=avg
        )
        if not avg:
            loss_r_img = loss_r_img.reshape(T, B)

        # log q(y | o) - image model
        loss_e_img = gaussian_log_prob(
            pack.y_mu_r, pack.y_log_sigma_r, y_r, use_torch=True
        )

        # log p(y | z) - state space model
        loss_r_ssm = -gaussian_log_prob(pack.y_mu_s, pack.y_cov_s, y_r)

        # log p(y | z) - prediction instead of smoothing
        z0_mu = pack.z_mu_s[0]
        z0_cov = pack.z_cov_s[0]
        if self._cond_channels > 0:
            cond = batch_o[0, :, -self._cond_channels :, 0, 0]
        else:
            cond = None

        if avg:
            loss_e_img = torch.sum(loss_e_img) / (T * B)
            loss_r_ssm = torch.sum(loss_r_ssm) / (T * B)

        burn_in_coeff = min(1.0, iteration / self._kf._burn_in)

        # D_KL(q(z | y) || p(z)) - state space model
        if self._cond_channels > 0:
            cond = batch_o[..., -self._cond_channels :, 0, 0]  # conditional context
        else:
            cond = None
        anneal_coeff = min(1.0, iteration / self._kf._dkl_anneal_iter)
        loss_dkl_ssm = (
            self._kf._beta
            * anneal_coeff
            * self._kf.kl_loss(
                pack.z_mu_s, pack.z_cov_s, batch_t, batch_u, cond=cond, avg=avg
            )
        )

        # pred loss over images o or latent states z
        if not self._kf._z_pred:
            loss_r_img = self._kf._alpha * loss_r_img

            log_dsd_p = self.predict(z0_mu, z0_cov, batch_t, batch_u, cond=cond)
            loss_p_img, _ = self._image_model.loss(
                batch_o, pack.y_mu_r, pack.y_log_sigma_r, log_dsd_p, avg=avg
            )  # pack args are dummy, only used for computation of unused kl term

            y_mu_p, y_cov_p = self._kf.predict(
                z0_mu, z0_cov, batch_t, batch_u, cond=cond, return_hidden=False,
            )
            loss_p_img = (1 - self._kf._alpha) * burn_in_coeff * loss_p_img
            loss_r_img = loss_r_img + loss_p_img

        else:
            z_mu_p, z_cov_p = self._kf.predict(
                z0_mu, z0_cov, batch_t, batch_u, cond=cond, return_hidden=True,
            )
            loss_r_ssm_p = -gaussian_log_prob(z_mu_p, z_cov_p, pack.z_mu_s)
            loss_r_ssm = (
                self._kf._alpha * loss_r_ssm
                + burn_in_coeff * (1 - self._kf._alpha) * loss_r_ssm_p
            )
            if avg:
                loss_r_ssm = loss_r_ssm.mean()

        if return_components:
            return loss_r_img, loss_e_img, loss_r_ssm, loss_dkl_ssm
        else:
            return loss_r_img + loss_e_img + loss_r_ssm + loss_dkl_ssm

    def log(self, buddy: Buddy, viz: VisData, filter_length: int = 1) -> None:
        """Logs data during training.

        Logged Information
        ------------------
        - Scalar loss components
        - Image sequence filtering/smoothing/prediction vs. ground truth
        - Latent manifold visualization
        - Correlation plots assuming affine relation between z and simulated state data

        Also logs loss components of the viz batch as scalars.

        Parameters
        ----------
        See parent class.
        """
        assert isinstance(viz, VisDataIMG)

        # ---scalar logging--- #
        loss_r_img, loss_er_img, loss_r_ssm, loss_dkl_ssm = self.loss(
            viz.t[:filter_length],
            viz.y[:filter_length],
            viz.u[:filter_length],
            1e7,
            return_components=True,
        )
        log_scalars(
            buddy,
            {
                "Validation_R_IMG-Loss": loss_r_img.item(),
                "Validation_ER_IMG-Loss": loss_er_img.item(),
                "Validation_R_SMM-Loss": loss_r_ssm.item(),
                "Validation_KL_SSM-Loss": loss_dkl_ssm.item(),
            },
            scope="Validation_KVAE",
        )

        # ---image sequences--- #
        assert (
            viz.t.shape[0] > filter_length
        ), "Visualization length must be longer than filter length."
        predict_length = min(2 * filter_length, viz.t.shape[0] - filter_length)

        total_len = filter_length + predict_length
        p_img, _ = _img_seq(self, viz, filter_length, predict_length)
        log_image(buddy, p_img, "image_trajectories", scope="sequence_visualizations")

        # ---latent visualization--- #
        # smoothed samples
        pack_fs, _ = self(viz.t[:total_len], viz.y[:total_len], viz.u[:total_len])
        z_mu_s = pack_fs.z_mu_s
        z_cov_s = pack_fs.z_cov_s
        z_samp_s = reparameterize_gauss(z_mu_s, z_cov_s).cpu().numpy()

        # predicted samples
        z0_mu_p = z_mu_s[0]
        z0_cov_p = z_cov_s[0]
        if self._cond_channels > 0:
            cond = viz.y[0, :, -self._cond_channels :, 0, 0]  # conditional context
        else:
            cond = None
        z_mu_p, z_cov_p = self._kf.predict(
            z0_mu_p,
            z0_cov_p,
            viz.t[:total_len],
            viz.u[:total_len],
            cond=cond,
            return_hidden=True,
        )
        z_samp_p = reparameterize_gauss(z_mu_p, z_cov_p).cpu().numpy()

        if self._kf.cell._latent_dim == 2:
            p_img_s, _ = _2d_latent_viz(z_samp_s, "Smoothed Latent Trajectories")
            p_img_p, _ = _2d_latent_viz(z_samp_p, "Predicted Latent Trajectories")

            ftitle = "smoothed_latent_trajectories"
            log_image(buddy, p_img_s, f"{ftitle}", scope="latent_viz")
            ftitle = "predicted_latent_trajectories"
            log_image(buddy, p_img_p, f"{ftitle}", scope="latent_viz")

        elif self._kf.cell._latent_dim == 3:
            p_img_s, _ = _3d_latent_viz(z_samp_s, "Smoothed Latent Trajectories")
            p_img_p, _ = _3d_latent_viz(z_samp_p, "Predicted Latent Trajectories")

            ftitle = "smoothed_latent_trajectories"
            log_image(buddy, p_img_s, f"{ftitle}", scope="latent_viz")
            ftitle = "predicted_latent_trajectories"
            log_image(buddy, p_img_p, f"{ftitle}", scope="latent_viz")

        # ---correlation plots--- #
        p_img_list, _ = _corr_plots(
            self, viz, filter_length, 2 * filter_length, z_samp_s, z_samp_p
        )
        filename_labels = ["x", "y", "theta", "dx", "dy", "dtheta"]
        for i in range(len(p_img_list)):
            p_img = p_img_list[i]
            log_image(
                buddy,
                p_img,
                f"correlation_plot_{filename_labels[i]}",
                scope="correlation_viz",
            )

    def eval_loss(self, viz: VisData, filt_points: int, pred_points: int) -> None:
        """Prints evaluation losses for the model.

        See parent method.
        """
        assert isinstance(viz, VisDataIMG)
        assert filt_points + pred_points <= len(viz.t)

        # filtering and prediction time/data
        t_filt = viz.t[:filt_points]
        o_filt = viz.y[:filt_points]
        u_filt = viz.u[:filt_points]
        t_pred = viz.t[(filt_points - 1) : (filt_points + pred_points - 1)]
        o_pred = viz.y[(filt_points - 1) : (filt_points + pred_points - 1)]
        u_pred = viz.u[(filt_points - 1) : (filt_points + pred_points - 1)]

        T, B = o_pred.shape[:2]

        # filtering
        pack, _ = self(t_filt, o_filt, u_filt)

        # prediction
        z0_mu_p = pack.z_mu_s[-1]
        z0_cov_p = pack.z_cov_s[-1]
        if self._cond_channels > 0:
            _o_pred = viz.y[0, :, : -self._cond_channels, 0, 0]
            cond = viz.y[0, :, -self._cond_channels :, 0, 0]  # conditional context
        else:
            _o_pred = o_pred
            cond = None

        # computing L2 loss
        log_dsd_p = self.predict(
            z0_mu_p, z0_cov_p, t_pred, u_pred, cond=cond
        )  # predicted log-softmax distribution
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
            f"L2 Loss: {loss_l2.item():.5f} \t"
            f"NLL: {nll.item():.3f}"
        )
        return nll.item(), loss_l2.item()

    def summary_plot(self, viz: VisData, name: str, debug: bool = False) -> None:
        """Produces and saves summary plots. Will produce same plots as log().

        See parent class.
        """
        assert isinstance(viz, VisDataIMG)

        # create summary plot directory corresponding to this experiment
        # > stackoverflow.com/questions/273192
        DIR_NAME = "summary_plots/" + name
        Path(DIR_NAME).mkdir(parents=True, exist_ok=True)

        filter_length = 25
        pred_length = 50

        # ---image sequences--- #
        _, seq_fig = _img_seq(self, viz, filter_length, pred_length)
        seq_fig.savefig(DIR_NAME + "/img_seq")

        # ---latent viz---#
        t_filt = viz.t[:filter_length]
        y_filt = viz.y[:filter_length]
        u_filt = viz.u[:filter_length]

        t_pred = viz.t[(filter_length - 1) : (filter_length + pred_length - 1)]
        y_pred = viz.y[(filter_length - 1) : (filter_length + pred_length - 1)]
        u_pred = viz.u[(filter_length - 1) : (filter_length + pred_length - 1)]

        pack_fs, _ = self(t_filt, y_filt, u_filt)
        z_mu_s = pack_fs.z_mu_s
        z_cov_s = pack_fs.z_cov_s
        z_samp_s = reparameterize_gauss(z_mu_s, z_cov_s).cpu().numpy()

        # predicted latent sequence
        z0_mu_p = z_mu_s[-1]
        z0_cov_p = z_cov_s[-1]
        if self._cond_channels > 0:
            cond = y_filt[0, :, -self._cond_channels :, 0, 0]  # conditional context
        else:
            cond = None
        z_mu_p, z_cov_p = self._kf.predict(
            z0_mu_p, z0_cov_p, t_pred, u_pred, cond=cond, return_hidden=True
        )
        z_samp_p = reparameterize_gauss(z_mu_p, z_cov_p).cpu().numpy()

        z_samp = np.concatenate((z_samp_s, z_samp_p), axis=0)

        if self._kf.cell._latent_dim == 2:
            _, latent_viz_fig = _2d_latent_viz(z_samp, "Latent Trajectories")
            latent_viz_fig.savefig(DIR_NAME + "/latent_viz")

        elif self._kf.cell._latent_dim == 3:
            _, latent_viz_fig = _3d_latent_viz(z_samp, "Latent Trajectories")
            latent_viz_fig.savefig(DIR_NAME + "/latent_viz")

        # ---pred loss per step--- #
        if self._cond_channels > 0:
            cond_all = cond.unsqueeze[0].repeat(z_mu_p.shape[0], 1, 1)
        else:
            cond_all = None
        log_dsd_p = self.latent_to_observation(z_mu_p, z_cov_p, cond=cond_all)
        bce_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        o_pred_one_hot = F.one_hot(
            (y_pred.squeeze(2) * (self._image_model.pixel_res - 1)).long()
        )
        o_pred_one_hot = o_pred_one_hot.float().permute(0, 1, 4, 2, 3) / (
            self._image_model.pixel_res - 1
        )
        _losses = (
            torch.sum(bce_criterion(log_dsd_p, o_pred_one_hot), dim=(1, 2, 3, 4))
            / o_pred_one_hot.shape[1]
        )  # shape=(T)
        losses = _losses.cpu().numpy()

        fig = plt.figure()
        plt.plot(losses)
        plt.xlabel("Prediction Step")
        plt.ylabel("Loss")
        plt.title("Loss Per Prediction Step")
        plt.savefig(DIR_NAME + "/loss_step")
        plt.close(fig)

        # ---correlation plots--- #
        # times = viz.t[: filter_length + pred_length]
        # img_data = viz.y[: filter_length + pred_length]
        # inputs = viz.u[: filter_length + pred_length]

        # pack_fs, _ = self(times, img_data, inputs)
        # z_mu_s = pack_fs.z_mu_s
        # z_cov_s = pack_fs.z_cov_s
        # z_samp_s = reparameterize_gauss(z_mu_s, z_cov_s).cpu().numpy()

        # z0_mu_p = z_mu_s[0]
        # z0_cov_p = z_cov_s[0]
        # z_mu_p, z_cov_p = self._kf.predict(
        #     z0_mu_p, z0_cov_p, times, inputs, cond=cond, return_hidden=True
        # )
        # z_samp_p = reparameterize_gauss(z_mu_p, z_cov_p).cpu().numpy()

        # _, corr_fig_list = _corr_plots(
        #     self, viz, filter_length, pred_length, z_samp_s, z_samp_p
        # )
        # filename_labels = ["x", "y", "theta", "dx", "dy", "dtheta"]
        # for i in range(len(corr_fig_list)):
        #     corr_fig = corr_fig_list[i]
        #     corr_fig.savefig(DIR_NAME + "/corr_" + filename_labels[i])


@dataclass(frozen=True)
class KVAEConfig(EstimatorConfig):
    """KVAE specific configuration parameters."""

    latent_obs_dim: int
    kf_estimator_config: KalmanEstimatorConfig
    dataset: ImageDynamicDatasetConfig  # additional hint for requiring an image dataset
    use_dsd: bool = True
    pixel_res: int = 256
    image_model: Optional[nn.Module] = None
    image_model_path: Optional[Tuple[str, bool]] = None
    cond_channels: int = 0

    def create(self) -> KVAE:
        """See parent."""
        C, H, W = self.dataset.obs_shape

        class DummyDataset:
            obs_shape = (self.latent_obs_dim,)

        kf_estimator_config = replace(self.kf_estimator_config, dataset=DummyDataset())
        return KVAE(
            self,
            H,
            W,
            C,
            self.pixel_res,
            self.use_dsd,
            kf_estimator_config.create(),
            image_model=self.image_model,
            image_model_path=self.image_model_path,
            cond_channels=self.cond_channels,
        )


# --------------- #
# LOGGING HELPERS #
# --------------- #


def _prep_axis(ax: axes.Axes, label: str) -> None:
    """Helper for KVAE plots."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylabel(
        f"{label}", rotation="horizontal", ha="right", va="center", fontsize=14
    )


def _img_seq(
    model: Estimator,
    viz: VisDataIMG,
    filter_length: int,
    pred_length: int,
    _idx: Optional[int] = None,
) -> Tuple[np.ndarray, matplotlib.figure.Figure]:
    """Helper for image sequence plotting."""
    # apply the learning curriculum to visualization due to numerical instability
    tot_len = filter_length + pred_length
    times = viz.t[:tot_len]
    img_data = viz.y[:tot_len]
    inputs = viz.u[:tot_len]
    T, B, _, img_size = img_data.shape[:4]

    # select random viz image - index into this slice for all image sequence viz
    if _idx is None:
        idx = np.random.randint(B)
    else:
        idx = _idx
    img_data = img_data[:, idx : (idx + 1)]
    inputs = inputs[:, idx : (idx + 1)]

    if isinstance(model, KVAE):
        # filtering and smoothing, "_fs" denotes filter-smoother
        _pack_fs, _ = model(times, img_data, inputs)
        img_filt = model._image_model.sample_img(_pack_fs.log_dsd_f)
        img_smth = model._image_model.sample_img(_pack_fs.log_dsd_s)

        # filtering to initialize prediction frames, "_p" denotes prediction
        _pack_p, _ = model(
            times[:filter_length], img_data[:filter_length], inputs[:filter_length]
        )
        img_pred_1 = model._image_model.sample_img(_pack_p.log_dsd_s)

        # predicting image frames after given frames
        z0_mu = _pack_p.z_mu_s[-1]
        z0_cov = _pack_p.z_cov_s[-1]
    else:
        # Hack for allowing PlaNet model to also visualize images.
        z0 = model.get_initial_hidden_state(img_data.shape[1])
        z_mu, _ = model(times, img_data, inputs, z0, return_hidden=True)
        img_filt = model._image_model.sample_img(
            model.latent_to_observation(z_mu, None)
        )
        img_smth = model._image_model.sample_img(torch.zeros_like(img_filt))

        z_mu, z_cov = model(
            times[:filter_length],
            img_data[:filter_length],
            inputs[:filter_length],
            z0,
            return_hidden=True,
        )
        img_pred_1 = model._image_model.sample_img(
            model.latent_to_observation(z_mu, None)
        )
        z0_mu = z_mu[-1]
        z0_cov = z_cov[-1]

    # ignore first prediction, use last filter sample instead
    u_pred = inputs[(filter_length - 1) :]
    if model._cond_channels > 0:
        cond = viz.y[
            0, idx : (idx + 1), -model._cond_channels :, 0, 0
        ]  # conditional context
    else:
        cond = None
    log_dsd_2 = model.predict(
        z0_mu, z0_cov, times[(filter_length - 1) :], u_pred, cond=cond
    )[1:]
    img_pred_2 = model._image_model.sample_img(log_dsd_2)

    img_pred = torch.cat([img_pred_1, img_pred_2], dim=0)

    # Visualize VAE reconstructions
    reconstructed_img_data = model._image_model.reconstruct(
        img_data.squeeze(1)
    ).unsqueeze(1)

    # canvases for plotting - black lines separating individual frames
    canvas_rows = 5
    canvas_frames = [
        np.zeros((img_size + 2, T * (img_size + 1) + 1)) for _ in range(canvas_rows)
    ]
    for t in range(T):
        for idx, img_plot_data in enumerate(
            [img_data, reconstructed_img_data, img_filt, img_smth, img_pred]
        ):
            frame = img_plot_data[t, 0, 0].cpu().detach().numpy()
            # adding to canvas
            start_col = t * (img_size + 1) + 1
            end_col = (t + 1) * (img_size + 1)
            canvas_frames[idx][1:-1, start_col:end_col] = frame

    # convert canvas_for prediction frames to RGB for plotting red line at the end of given data
    # color channel is in dim=-1 for imshow()
    canvas_frames[-1] = np.tile(np.expand_dims(canvas_frames[-1], axis=-1), (1, 1, 3))
    divide_col = filter_length * (img_size + 1)
    canvas_frames[-1][1:-1, divide_col, 0] = 1.0

    with ph.plot_context(sp_shape=(canvas_rows, 1)) as (fig, axs):
        # helpful: stackoverflow.com/questions/41071947/
        # extra space for ylabels, vertical spacing is tight
        fig.set_size_inches((T + 1.5) / 1.5, 4 / 1.5)
        gs = gridspec.GridSpec(canvas_rows, 1, hspace=0.0)

        for i in range(canvas_rows):
            axs[i] = plt.subplot(gs[i, 0])

        _prep_axis(axs[0], "Data")
        _prep_axis(axs[1], "Reconstruction")
        _prep_axis(axs[2], "Filter")
        _prep_axis(axs[3], "Smoother")
        _prep_axis(axs[4], "Prediction")

        # add frame numbering to bottom axis
        xlabels = [i + 1 for i in range(T)]
        xticks = [(t + 0.5) * (img_size + 1) for t in range(T)]
        axs[-1].set_xticks(xticks)
        axs[-1].set_xticklabels(xlabels, fontsize=14)
        axs[-1].tick_params(axis="both", which="both", length=0)  # hide ticks

        # plot and format as image
        for i in range(canvas_rows - 1):
            axs[i].imshow(canvas_frames[i], cmap="gray", vmin=0.0, vmax=1.0)
        axs[-1].imshow(canvas_frames[4], vmin=0.0, vmax=1.0)

        plt.tight_layout()
        p_img = ph.plot_as_image(fig)

    return p_img, fig


def _2d_latent_viz(
    z_samp: torch.Tensor, title: str
) -> Tuple[np.ndarray, matplotlib.figure.Figure]:
    """Helper for 2D latent viz plots."""
    B = z_samp.shape[1]

    with ph.plot_context(sp_shape=(1, 1)) as (fig, ax):
        # each traj different color
        # > useful: stackoverflow.com/questions/12236566
        colors = plt.cm.rainbow(np.linspace(0, 1, B))
        for b in range(B):
            z_1 = z_samp[:, b, 0]
            z_2 = z_samp[:, b, 1]

            color = colors[b]
            ax.plot(z_1, z_2, color=color)
            ax.scatter(z_1[0], z_2[0], color=color, marker="o")
            ax.scatter(z_1[-1], z_2[-1], color=color, marker="x")

        # axis settings
        circ_legend = mlines.Line2D(
            [], [], color="k", marker="o", linestyle="None", label="start"
        )
        x_legend = mlines.Line2D(
            [], [], color="k", marker="x", linestyle="None", label="end"
        )
        ax.legend(
            handles=[circ_legend, x_legend], bbox_to_anchor=(1.01, 1), loc="upper left",
        )
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$")
        ax.set_title(f"{title}")
        plt.axis("square")
        ax.grid(False)

        p_img = ph.plot_as_image(fig)
    return p_img, fig


def _3d_latent_viz(
    z_samp: torch.Tensor, title: str
) -> Tuple[np.ndarray, matplotlib.figure.Figure]:
    """Helper for 3D latent viz plots."""
    B = z_samp.shape[1]

    with ph.plot3d_context() as (fig, ax):
        # each traj different color
        # > useful: stackoverflow.com/questions/12236566
        colors = plt.cm.rainbow(np.linspace(0, 1, B))
        for b in range(B):
            z_1 = z_samp[:, b, 0]
            z_2 = z_samp[:, b, 1]
            z_3 = z_samp[:, b, 2]

            color = colors[b]
            ax.plot(z_1, z_2, z_3, color=color)
            ax.scatter(z_1[0], z_2[0], z_3[0], color=color, marker="o")
            ax.scatter(z_1[-1], z_2[-1], z_3[-1], color=color, marker="x")

        # axis settings
        circ_legend = mlines.Line2D(
            [], [], color="k", marker="o", linestyle="None", label="start"
        )
        x_legend = mlines.Line2D(
            [], [], color="k", marker="x", linestyle="None", label="end"
        )
        ax.legend(
            handles=[circ_legend, x_legend], bbox_to_anchor=(1.01, 1), loc="upper left",
        )
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$")
        ax.set_zlabel(r"$z_3$")
        ax.set_title(f"{title}")
        ax.grid(False)

        # aspect ratio stuff
        # > see: github.com/matplotlib/matplotlib/issues/17172
        xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
        XYZlim = [min(xyzlim[0]), max(xyzlim[1])]
        ax.set_xlim3d(XYZlim)
        ax.set_ylim3d(XYZlim)
        ax.set_zlim3d(XYZlim)
        ax.set_box_aspect((1, 1, 1))

        p_img = ph.plot_as_image(fig)
    return p_img, fig


def _corr_plots(
    kvae: KVAE,
    viz: VisDataIMG,
    filter_length: int,
    pred_length: int,
    z_samp_s: np.array,
    z_samp_p: np.array,
) -> Tuple[List[np.ndarray], List[matplotlib.figure.Figure]]:
    """Helper for correlation plots."""
    tot_len = filter_length + pred_length

    # recover position/velocity ground truth
    pos_vel_data = viz.pv
    x_data = pos_vel_data[:tot_len, :, 0]
    y_data = pos_vel_data[:tot_len, :, 1]
    dx_data = pos_vel_data[:tot_len, :, 2]
    dy_data = pos_vel_data[:tot_len, :, 3]

    # get angle data from cartesian data
    th_data = torch.atan2(x_data, -y_data)
    denom = x_data ** 2.0 + y_data ** 2.0
    dth_data = (-y_data * dx_data + x_data * dy_data) / denom

    # use OLS to compute affine map from z to data
    data_list = [x_data, y_data, dx_data, dy_data, th_data, dth_data]
    C, C_p, X = _ols_helper(data_list, z_samp_s, z_samp_p, kvae._device)

    # plotting predictions vs. ground truth
    plot_labels = [
        "x",
        "y",
        r"$\theta$",
        r"$\dot{x}$",
        r"$\dot{y}$",
        r"$\dot{\theta}$",
    ]

    # plot list
    p_img_list = []
    fig_list = []

    for i in range(C.shape[-1]):
        with ph.plot_context(sp_shape=(1, 1)) as (fig, ax):
            colors = plt.cm.Blues(np.linspace(0, 1, C.shape[0]))
            ax.scatter(C[:, i], C_p[:, i], c=colors, marker="o")

            # print coeffs with title
            # > stackoverflow.com/questions/5326112
            ax.set_title(
                f"Correlation: {plot_labels[i]}\n"
                + f"OLS Coeffs: {list(np.around(X[:, i], 2))}"
            )
            ax.set_xlabel(f"True {plot_labels[i]}")
            ax.set_ylabel(f"Predicted {plot_labels[i]}")

            # square the axis limits
            xylim = np.array([ax.get_xlim(), ax.get_ylim()]).T
            XYlim = [min(xylim[0]), max(xylim[1])]
            ax.set_xlim(XYlim)
            ax.set_ylim(XYlim)
            ax.set_aspect("equal")

            p_img = ph.plot_as_image(fig)
            p_img_list.append(p_img)
            fig_list.append(fig)

    return p_img_list, fig_list


def _ols_helper(
    data_list: List[torch.Tensor],
    z_samp_s: np.array,
    z_samp_p: np.array,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper function for doing OLS for pendulum data."""
    # setting up OLS matrices for linear regression: AX = C
    A = data_list[0].new_tensor(z_samp_s.reshape(-1, z_samp_s.shape[-1]))
    A = torch.cat([A, torch.ones(A.shape[0], 1, device=A.device)], dim=-1)
    C = torch.cat([data.reshape(-1, 1) for data in data_list], dim=-1)
    X = torch.lstsq(C, A)[0][: A.shape[-1]]  # columns contain regression coeffs

    # use coefficients to predict x, y, th, dx, dy, dth
    A_p = A.new_tensor(z_samp_p.reshape(-1, z_samp_p.shape[-1]))
    A_p = torch.cat([A_p, torch.ones(A_p.shape[0], 1, device=A.device)], dim=-1)
    C_p = A_p @ X
    return to_numpy(C), to_numpy(C_p), to_numpy(X)


def histogram_plots(
    model: Estimator, viz: VisDataIMG, filt_points: int, pred_points: int
) -> None:
    """Save a sequence of plots visualizaing the histograms at each timestep."""
    assert filt_points + pred_points <= len(viz.t)

    # filtering and prediction time/data
    t_filt = viz.t[:filt_points]
    o_filt = viz.y[:filt_points]
    u_filt = viz.u[:filt_points]
    t_pred = viz.t[(filt_points - 1) : (filt_points + pred_points - 1)]
    o_pred = viz.y[(filt_points - 1) : (filt_points + pred_points - 1)]
    u_pred = viz.u[(filt_points - 1) : (filt_points + pred_points - 1)]

    x_y_loc_pred = viz.np_pv[
        (filt_points - 1) : (filt_points + pred_points - 1), :, 0:2
    ]
    # convert x_y to pixel space
    r = 0.6
    gap = 4.0
    _img_size = 16
    pos_min = -2.0

    x = -x_y_loc_pred[..., 1:2]  # swap x and y in image space
    y = x_y_loc_pred[..., 0:1]

    x_px = ((_img_size - 1) / gap) * (x - pos_min)
    y_px = ((_img_size - 1) / gap) * (y - pos_min)
    r_px = ((_img_size - 1) / gap) * r + 1

    T, B = o_pred.shape[:2]

    # filtering
    pack, _ = model(t_filt, o_filt, u_filt)

    # prediction
    z0_mu_p = pack.z_mu_s[-1]
    z0_cov_p = pack.z_cov_s[-1]
    if model._cond_channels > 0:
        _o_pred = viz.y[0, :, : -model._cond_channels, 0, 0]
        cond = viz.y[0, :, -model._cond_channels :, 0, 0]  # conditional context
    else:
        _o_pred = o_pred
        cond = None

    # computing L2 loss
    log_dsd_p = model.predict(
        z0_mu_p, z0_cov_p, t_pred, u_pred, cond=cond
    )  # predicted log-softmax distribution
    oo_quant = (_o_pred.squeeze(2) * (model._image_model.pixel_res - 1)).long()

    def plot_hist_with_true(
        dsd: torch.Tensor,
        true: torch.Tensor,
        name: str,
        x_loc: int,
        y_loc: int,
        size=(16, 16),
    ) -> None:
        """Save the histogram at the region around the specified x, y location.

        The true pixel value at each given location is also plotted.
        """
        C, H, W = dsd.shape
        with ph.plot_context(sp_shape=(r_px * 2 + 1, r_px * 2 + 1), size=size) as (
            fig,
            axs,
        ):
            for h in range(x_loc - r_px, x_loc + r_px + 1):
                for w in range(y_loc - r_px, y_loc + r_px + 1):
                    axs[h - (x_loc - r_px), w - (y_loc - r_px)].bar(
                        list(range(C)), dsd[:, h, w].detach().cpu().numpy(), width=5,
                    )
                    axs[h - (x_loc - r_px), w - (y_loc - r_px)].bar(
                        list(range(C)),
                        true[:, h, w].detach().cpu().numpy() / 9,
                        width=5,
                    )
                    axs[h - (x_loc - r_px), w - (y_loc - r_px)].set_yticklabels([])
                    axs[h - (x_loc - r_px), w - (y_loc - r_px)].set_xticklabels([])
            # p_img = ph.plot_as_image(fig)
            fig.savefig(name, edgecolor="r", bbox_inches="tight")
            print(name)

    def plot_hist_with_true_only(
        true: torch.Tensor, name, x_loc: int, y_loc: int, size=(16, 16)
    ) -> None:
        """Save the histogram at the region around the specified x, y location."""
        C, H, W = true.shape
        with ph.plot_context(sp_shape=(r_px * 2 + 1, r_px * 2 + 1), size=size) as (
            fig,
            axs,
        ):
            for h in range(x_loc - r_px, x_loc + r_px + 1):
                for w in range(y_loc - r_px, y_loc + r_px + 1):
                    axs[h - (x_loc - r_px), w - (y_loc - r_px)].bar(
                        list(range(C)),
                        true[:, h, w].detach().cpu().numpy() / 9,
                        width=5,
                        color="tab:orange",
                    )
                    axs[h - (x_loc - r_px), w - (y_loc - r_px)].set_yticklabels([])
                    axs[h - (x_loc - r_px), w - (y_loc - r_px)].set_xticklabels([])
            fig.savefig(name, edgecolor="r", bbox_inches="tight")

    def _prep_axis(ax) -> None:
        """Remove ticks and labels from the axis."""
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def plot_pend_with_square(
        gray_img: torch.Tensor, x_loc: int, y_loc: int, name: str
    ) -> None:
        """Save the pendulum image with a bounding box at the given location."""
        canvas_frame = np.zeros((16, 16, 3))
        for i in range(3):
            canvas_frame[:, :, i] = gray_img.detach().cpu().numpy()

        with ph.plot_context(size=(16, 16)) as (fig, ax):
            _prep_axis(ax)
            ax.imshow(canvas_frame, vmin=0.0, vmax=1.0)
            rect = patches.Rectangle(
                (y_loc - r_px, x_loc - r_px),
                2 * r_px,
                2 * r_px,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
            fig.savefig(name, bbox_inches="tight")

    for p in ["tmp/true", "tmp/hist", "tmp/hist_true", "tmp/gen"]:
        Path(p).mkdir(parents=True, exist_ok=True)

    plot_batch_idx = 0
    for idx, (log_dsd_t, quant, ooo_pred, x_loc, y_loc) in enumerate(
        zip(
            log_dsd_p[:, plot_batch_idx, ...],
            oo_quant[:, plot_batch_idx, ...],
            _o_pred[:, plot_batch_idx, ...],
            x_px[:, plot_batch_idx],
            y_px[:, plot_batch_idx],
        )
    ):
        assert len(log_dsd_t.shape) == 3
        name = f"tmp/hist/img_seq-{idx:03}.png"
        true = torch.nn.functional.one_hot(quant).transpose(-1, -3).transpose(-1, -2)
        plot_hist_with_true(torch.exp(log_dsd_t), true, name, x_loc[0], y_loc[0])
        name = f"tmp/hist_true/img_seq-{idx:03}.png"
        plot_hist_with_true_only(true, name, x_loc[0], y_loc[0])
        sampled_img = model._image_model.sample_img(log_dsd_t)

        name = f"tmp/gen/gen_seq-{idx:03}.png"
        plot_pend_with_square(sampled_img, x_loc[0], y_loc[0], name)
        name = f"tmp/true/gen_seq_true-{idx:03}.png"
        plot_pend_with_square(quant.unsqueeze(0) / 255.0, x_loc[0], y_loc[0], name)
