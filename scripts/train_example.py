from collections import namedtuple

from fannypack.utils import pdb_safety_net
from torch import nn as nn

from dynamics_learning.custom.lr_functions import lr1
from dynamics_learning.data.datasets import (
    DynamicsDatasetType,
    DynamicSystemDatasetConfig,
)
from dynamics_learning.networks.estimator import EstimatorConfig
from dynamics_learning.networks.kalman.ekf import EKFEstimatorConfig
from dynamics_learning.training.configs import ExpConfig
from dynamics_learning.training.experiments import train

model_config: EstimatorConfig
exp_config: ExpConfig
pdb_safety_net()


# Configure hyper parameters
hyperparameter_defaults = dict(batch_size=8, learning_rate=5e-3, epochs=5, latent_dim=2)

HyperParameterConfig = namedtuple(
    "HyperParameterConfig", list(hyperparameter_defaults.keys())
)
hy_config = HyperParameterConfig(**hyperparameter_defaults)

# VDP settings
dataset_config = DynamicSystemDatasetConfig(
    traj_len=21,
    num_trajectories=10000,  # DEBUG
    num_viz_trajectories=25,
    system=DynamicsDatasetType.VDP,
    policy=None,
    dt=0.05,
)

# EKF settings
model_config = EKFEstimatorConfig(
    is_smooth=True,
    latent_dim=hy_config.latent_dim,
    ctrl_dim=0,
    dataset=dataset_config,
    dyn_hidden_units=32,
    dyn_layers=3,
    dyn_nonlinearity=nn.Softplus(beta=2, threshold=20),
    obs_hidden_units=32,
    obs_layers=3,
    obs_nonlinearity=nn.Softplus(beta=2, threshold=20),
    is_continuous=False,
    ramp_iters=100,
    burn_in=100,
    dkl_anneal_iter=1000,
    alpha=0.5,
    beta=1.0,
    atol=1e-9,  # default: 1e-9
    rtol=1e-7,  # default: 1e-7
    z_pred=False,
)

# experiment settings
exp_config = ExpConfig(
    model=model_config,
    ramp_iters=model_config.ramp_iters,
    batch_size=hy_config.batch_size,
    epochs=hy_config.epochs,
    log_iterations_simple=10,
    log_iterations_images=model_config.ramp_iters,
    base_learning_rate=hy_config.learning_rate,
    learning_rate_function=lr1,
)
train(exp_config)  # train the model
