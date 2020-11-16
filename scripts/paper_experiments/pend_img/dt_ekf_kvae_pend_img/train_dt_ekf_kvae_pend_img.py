from collections import namedtuple

from fannypack.utils import pdb_safety_net
from torch import nn as nn

from dynamics_learning.custom.lr_functions import lr5
from dynamics_learning.data.datasets import (
    DummyDatasetConfig,
    ImageDynamicDatasetConfig,
    PendFixedDatasetConfig,
)
from dynamics_learning.networks.estimator import EstimatorConfig
from dynamics_learning.networks.image_models.kvae import KVAEConfig
from dynamics_learning.networks.kalman.ekf import EKFEstimatorConfig
from dynamics_learning.training.configs import ExpConfig
from dynamics_learning.training.experiments import train

model_config: EstimatorConfig
exp_config: ExpConfig
pdb_safety_net()


hyperparameter_defaults = dict(batch_size=4, learning_rate=1e-3, epochs=3, latent_dim=3)

HyperParameterConfig = namedtuple(
    "HyperParameterConfig", list(hyperparameter_defaults.keys())
)
hy_config = HyperParameterConfig(**hyperparameter_defaults)

# PEND_IMG settings
pend_fixed = PendFixedDatasetConfig(traj_len=33, num_viz_trajectories=10)
dataset_config = ImageDynamicDatasetConfig(
    traj_len=pend_fixed.traj_len,
    num_trajectories=10000,
    num_viz_trajectories=pend_fixed.num_viz_trajectories,
    system=pend_fixed,
    policy=None,
)

# EKF-KVAE settings
_model_config = EKFEstimatorConfig(
    is_smooth=True,
    latent_dim=hy_config.latent_dim,
    ctrl_dim=1,
    dataset=DummyDatasetConfig(),
    dyn_hidden_units=64,
    dyn_layers=3,
    dyn_nonlinearity=nn.Softplus(beta=2, threshold=20),
    obs_hidden_units=64,
    obs_layers=3,
    obs_nonlinearity=nn.Softplus(beta=2, threshold=20),
    is_continuous=False,
    ramp_iters=200,
    burn_in=100,
    dkl_anneal_iter=1000,
    alpha=0.5,
    beta=2.0,
    atol=1e-9,  # default: 1e-9
    rtol=1e-7,  # default: 1e-7
    z_pred=False,
)
model_config = KVAEConfig(
    latent_dim=_model_config.latent_dim,
    ctrl_dim=_model_config.ctrl_dim,
    dataset=dataset_config,
    latent_obs_dim=2,
    kf_estimator_config=_model_config,
)

# experiment settings
exp_config = ExpConfig(
    name="dt_ekf_kvae_pend_img",
    model=model_config,
    ramp_iters=(
        _model_config.ramp_iters if hasattr(_model_config, "ramp_iters") else 100
    ),
    batch_size=hy_config.batch_size,
    epochs=hy_config.epochs,
    log_iterations_simple=10,
    log_iterations_images=model_config.kf_estimator_config.ramp_iters,
    base_learning_rate=hy_config.learning_rate,
    learning_rate_function=lr5,
    gradient_clip_max_norm=None,
)
train(exp_config)
