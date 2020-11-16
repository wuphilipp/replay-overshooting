from collections import namedtuple

from fannypack.utils import pdb_safety_net
from torch import nn as nn

from dynamics_learning.custom.lr_functions import lr5
from dynamics_learning.data.datasets import MITPushDatasetConfig
from dynamics_learning.networks.estimator import EstimatorConfig
from dynamics_learning.networks.kalman.ekf import EKFEstimatorConfig
from dynamics_learning.training.configs import ExpConfig
from dynamics_learning.training.experiments import train

model_config: EstimatorConfig
exp_config: ExpConfig
pdb_safety_net()


hyperparameter_defaults = dict(
    batch_size=16, learning_rate=5e-3, epochs=200, latent_dim=8
)

HyperParameterConfig = namedtuple(
    "HyperParameterConfig", list(hyperparameter_defaults.keys())
)
hy_config = HyperParameterConfig(**hyperparameter_defaults)

dataset_config = MITPushDatasetConfig(
    traj_len=50,
    num_viz_trajectories=20,
    pixel_res=256,
    raw=True,
    cond=False,
    half_image_size=False,
    kloss_dataset=True,  # true for mit push
)

model_config = EKFEstimatorConfig(
    is_smooth=True,
    latent_dim=hy_config.latent_dim,
    ctrl_dim=6 if dataset_config.kloss_dataset else 8,  # DEBUG: these are in flux
    dataset=dataset_config,
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
    beta=1.0,
    atol=1e-9,  # default: 1e-9
    rtol=1e-7,  # default: 1e-7
    z_pred=True,
)


exp_config = ExpConfig(
    name="dt_ekf_push",
    model=model_config,
    ramp_iters=(
        model_config.ramp_iters if hasattr(model_config, "ramp_iters") else 100
    ),
    batch_size=hy_config.batch_size,
    epochs=hy_config.epochs,
    log_iterations_simple=10,
    log_iterations_images=100,
    base_learning_rate=hy_config.learning_rate,
    learning_rate_function=lr5,
    gradient_clip_max_norm=100,
)
train(exp_config)
