from dataclasses import replace
from typing import Dict

import numpy as np
import torch
from fannypack.utils import Buddy, to_numpy

from dynamics_learning.data.datasets import VisData
from dynamics_learning.networks.estimator import Estimator
from dynamics_learning.utils.net_utils import reparameterize_gauss
from dynamics_learning.utils.plot_utils import PlotHandler as ph

DIM_TO_NAME = {0: "x", 1: "y"}


def _log_timeseries(
    buddy: Buddy,
    viz: VisData,
    y_f: torch.Tensor,
    y_p: torch.Tensor,
    filter_length: int,
    dim: int,
    name: str,
) -> None:
    """Helper to log timeseries compared data.

    Parameters
    ----------
    buddy : Buddy
        Buddy helper.
    viz : VisData
        Visualization data.
    y_f : torch.Tensor, shape=(T, B, p)
        Filtered samples.
    y_p : torch.Tensor, shape=(T, B, p)
        Predicted samples.
    filter_length : int
        Number of points to filter over (rest are prediction baselines).
    dim : int
        Dimension to plot (x or y for 2D data).
    name : str
        Name of plot.
    """
    pred_len = len(y_p)
    with ph.plot_context(viz.plot_settings) as (fig, ax):
        ph.plot_timeseries_compare(
            [
                viz.np_t[:filter_length],
                viz.np_t[(filter_length - 1) : (pred_len + filter_length - 1)],
                viz.np_t[:filter_length],
                viz.np_t[(filter_length - 1) : (pred_len + filter_length - 1)],
            ],
            [
                viz.np_y[:filter_length, :, dim],
                viz.np_y[(filter_length - 1) : (pred_len + filter_length - 1), :, dim],
                y_f[..., dim],
                y_p[..., dim],
            ],
            DIM_TO_NAME[dim],
            style_list=["b--", "b-", "r--", "r-"],
            startmark_list=["bo", None, "ro", None],
            endmark_list=[None, "bx", None, "rx"],
        )
        p_img = ph.plot_as_image(fig)

    _name = f"{name}-prediction-trajectory-{DIM_TO_NAME[dim]}t"
    buddy.log_image(_name, p_img)


def _log_filter(
    buddy: Buddy,
    viz: VisData,
    y_mu_f: torch.Tensor,
    filter_length: int,
    dim: int,
    name: str,
) -> None:
    """Helper to log filter timeseries compared data.

    Parameters
    ----------
    See _log_timeseries().
    """
    with ph.plot_context(viz.plot_settings) as (fig, ax):
        ph.plot_timeseries_compare(
            [viz.np_t, viz.np_t[:filter_length]],
            [viz.np_y[..., dim], y_mu_f[..., dim]],
            DIM_TO_NAME[dim],
            style_list=["b-", "r-"],
            startmark_list=["bo", "ro"],
            endmark_list=["bx", "rx"],
        )
        p_img = ph.plot_as_image(fig)
    _name = f"{name}-{DIM_TO_NAME[dim]}"
    buddy.log_image(_name, p_img)


def log_scalars(buddy: Buddy, scalar_dict: Dict[str, float], scope=None):
    if scope is not None:
        buddy.log_scope_push(scope)
    for name, value in scalar_dict.items():
        buddy.log_scalar(name, value)
    if scope is not None:
        buddy.log_scope_pop(scope)


def log_image(buddy: Buddy, image: np.ndarray, name: str, scope=None):
    if scope is not None:
        buddy.log_scope_push(scope)
    buddy.log_image(name, image)
    if scope is not None:
        buddy.log_scope_pop(scope)


def log_basic(
    estimator: Estimator,
    buddy: Buddy,
    viz: VisData,
    filter_length: int = 1,
    smooth: bool = False,
    plot_means: bool = True,
    ramp_pred: bool = False,
) -> None:
    """Log basic visual information for Gaussian estimators with filter ONLY.

    Parameters
    ----------
    estimator : Estimator
        The estimator.
    buddy : Buddy
        Buddy helper for training.
    viz : VisData
        Visualization data.
    filter_length : int, default=1
        Length of data to provide for filtering during prediction runs.
    smooth : bool, default=False
        Flag indicating whether estimator should smooth.
    plot_means : bool, default=True
        Flag indicating whether to plot means in addition to the samples.
    ramp_pred : bool, default=False
        Flag indicating whether to ramp pred horizon. Used for pred visualizations early
        on in KF training when it is numerically unstable.
    """
    assert filter_length >= 1
    filter_length = min(filter_length, len(viz.np_t))
    data_var = 0.0  # variance of independent Gaussian injected noise

    # ramp the visualizations
    if ramp_pred and hasattr(estimator, "_ramp_iters"):
        it = buddy.optimizer_steps
        idx_p = min((it // estimator._ramp_iters) + filter_length + 1, len(viz.t))  # type: ignore
    else:
        idx_p = None

    # ---------- #
    # PREDICTION #
    # ---------- #

    # filtered portion
    z0 = estimator.get_initial_hidden_state(viz.y0.shape[0])
    noise = torch.randn_like(viz.y[0:filter_length]) * np.sqrt(data_var)
    z_mu_f, z_cov_f = estimator(
        viz.t[:filter_length],
        viz.y[:filter_length] + noise,
        viz.u[:filter_length],
        z0,
        return_hidden=True,
    )
    y_mu_f, y_cov_f = estimator(
        viz.t[:filter_length], viz.y[:filter_length] + noise, viz.u[:filter_length], z0,
    )

    # smooth if possible
    if smooth:
        z_mu_s, z_cov_s = estimator.get_smooth()  # type: ignore
        y_mu_s, y_cov_s = estimator.latent_to_observation(z_mu_s, z_cov_s)
    else:
        z_mu_s = z_mu_f
        z_cov_s = z_cov_f
        y_mu_s = y_mu_f
        y_cov_s = y_cov_f

    # predicting
    y_mu_p, y_cov_p = estimator.predict(
        z_mu_s[-1],
        z_cov_s[-1],
        viz.t[(filter_length - 1) : idx_p],
        viz.u[(filter_length - 1) : idx_p],
    )

    # sampling from observation distributions
    y_samp_s = to_numpy(reparameterize_gauss(y_mu_s, y_cov_s))
    y_samp_p = to_numpy(reparameterize_gauss(y_mu_p, y_cov_p))
    y_mu_s = to_numpy(y_mu_s)
    y_mu_p = to_numpy(y_mu_p)

    # log prediction vs. ground truth
    with buddy.log_scope("0_predict"):
        # plotting predictions samples versus ground truth
        with ph.plot_context(viz.plot_settings) as (fig, ax):
            if filter_length == 1:
                ph.plot_xy_compare(
                    [y_samp_p, viz.np_y[:, :, 0:2]],
                    style_list=["r-", "b-"],
                    startmark_list=["ro", "bo"],
                    endmark_list=["rx", "bx"],
                )
            else:
                ph.plot_xy_compare(
                    [
                        y_samp_p,
                        viz.np_y[(filter_length - 1) : idx_p, :, 0:2],
                        y_samp_s,
                        viz.np_y[:filter_length, :, 0:2],
                    ],
                    style_list=["r-", "b-", "r--", "b--"],
                    startmark_list=[None, None, "ro", "bo"],
                    endmark_list=["rx", "bx", None, None],
                )
            p_img = ph.plot_as_image(fig)
        buddy.log_image("samples-xy-trajectory", p_img)

        # plotting means versus ground truth
        if plot_means:
            with ph.plot_context(viz.plot_settings) as (fig, ax):
                if filter_length == 1:
                    ph.plot_xy_compare(
                        [y_mu_p, viz.np_y[:, :, 0:2]],
                        style_list=["r-", "b-"],
                        startmark_list=["ro", "bo"],
                        endmark_list=["rx", "bx"],
                    )
                else:
                    ph.plot_xy_compare(
                        [
                            y_mu_p,
                            viz.np_y[(filter_length - 1) : idx_p, :, 0:2],
                            y_mu_s,
                            viz.np_y[:filter_length, :, 0:2],
                        ],
                        style_list=["r-", "b-", "r--", "b--"],
                        startmark_list=[None, None, "ro", "bo"],
                        endmark_list=["rx", "bx", None, None],
                    )
                p_img = ph.plot_as_image(fig)
            buddy.log_image("means-xy-trajectory", p_img)

        # xy timeseries - samples and means
        _log_timeseries(buddy, viz, y_samp_s, y_samp_p, filter_length, 0, "samples")
        _log_timeseries(buddy, viz, y_samp_s, y_samp_p, filter_length, 1, "samples")
        if plot_means:
            _log_timeseries(buddy, viz, y_mu_s, y_mu_p, filter_length, 0, "means")
            _log_timeseries(buddy, viz, y_mu_s, y_mu_p, filter_length, 1, "means")

    # -------------- #
    # FILTERING ONLY #
    # -------------- #

    # sample traj image
    z0 = estimator.get_initial_hidden_state(viz.y0.shape[0])
    noise = np.sqrt(data_var) * torch.randn_like(viz.y)
    z_mu_f, z_cov_f = estimator(
        viz.t[:filter_length],
        viz.y[:filter_length] + noise[:filter_length],
        viz.u[:filter_length],
        z0,
        return_hidden=True,
    )

    # sampling from observation distributions
    y_mu_f, y_cov_f = estimator.latent_to_observation(z_mu_f, z_cov_f)
    y_samp_f = to_numpy(reparameterize_gauss(y_mu_f, y_cov_f))
    y_mu_f = to_numpy(y_mu_f)

    # xy filter plots
    with buddy.log_scope("1_filter-xy-trajectory"):
        # plotting samples vs. ground truth
        with ph.plot_context(viz.plot_settings) as (fig, ax):
            ph.plot_xy_compare([y_samp_f, viz.np_y[:, :, 0:2]])
            p_img = ph.plot_as_image(fig)
        buddy.log_image("samples-no-noise", p_img)

        with ph.plot_context(viz.plot_settings) as (fig, ax):
            ph.plot_xy_compare([y_samp_f, viz.np_y + to_numpy(noise)])
            p_img = ph.plot_as_image(fig)
        buddy.log_image("samples-meas-noise", p_img)

        # plotting means vs. ground truth
        if plot_means:
            with ph.plot_context(viz.plot_settings) as (fig, ax):
                ph.plot_xy_compare([y_mu_f, viz.np_y[:, :, 0:2]])
                p_img = ph.plot_as_image(fig)
            buddy.log_image("means-no-noise", p_img)

            with ph.plot_context(viz.plot_settings) as (fig, ax):
                ph.plot_xy_compare([y_mu_f, viz.np_y + to_numpy(noise)])
                p_img = ph.plot_as_image(fig)
            buddy.log_image("means-meas-noise", p_img)

    # time filter plots
    with buddy.log_scope("1_filter-t-trajectory"):
        viz_noise = replace(viz, np_y=viz.np_y + to_numpy(noise))

        # x,y samples no noise/with noise
        _log_filter(buddy, viz, y_samp_f, filter_length, 0, "samples-no-noise")
        _log_filter(buddy, viz, y_samp_f, filter_length, 1, "samples-no-noise")
        _log_filter(
            buddy, viz_noise, y_samp_f, filter_length, 0, "samples-meas-noise",
        )
        _log_filter(
            buddy, viz_noise, y_samp_f, filter_length, 1, "samples-meas-noise",
        )

        # x,y means no noise/with noise
        if plot_means:
            _log_filter(buddy, viz, y_mu_f, filter_length, 0, "means-no-noise")
            _log_filter(buddy, viz, y_mu_f, filter_length, 1, "means-no-noise")
            _log_filter(
                buddy, viz_noise, y_mu_f, filter_length, 0, "means-meas-noise",
            )
            _log_filter(
                buddy, viz_noise, y_mu_f, filter_length, 1, "means-meas-noise",
            )
