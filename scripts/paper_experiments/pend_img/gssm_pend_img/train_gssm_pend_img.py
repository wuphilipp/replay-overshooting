from collections import namedtuple

from fannypack.utils import pdb_safety_net

from dynamics_learning.custom.lr_functions import lr2
from dynamics_learning.data.datasets import (
    ImageDynamicDatasetConfig,
    PendFixedDatasetConfig,
)
from dynamics_learning.networks.baseline.ssm_baseline import GSSMBaselineConfig
from dynamics_learning.networks.estimator import EstimatorConfig
from dynamics_learning.training.configs import ExpConfig
from dynamics_learning.training.experiments import train

model_config: EstimatorConfig
exp_config: ExpConfig
pdb_safety_net()


hyperparameter_defaults = dict(
    batch_size=8, learning_rate=1e-3, epochs=80, latent_dim=3
)

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

model_config = GSSMBaselineConfig(
    latent_dim=hy_config.latent_dim,
    latent_obs_dim=8,
    hidden_units=64,
    ctrl_dim=1,
    dataset=dataset_config,
)

exp_config = ExpConfig(
    name="gssm_pend_img",
    model=model_config,
    ramp_iters=100,
    batch_size=hy_config.batch_size,
    epochs=hy_config.epochs,
    log_iterations_simple=10,
    log_iterations_images=100,
    base_learning_rate=hy_config.learning_rate,
    learning_rate_function=lr2,
    gradient_clip_max_norm=500,
)
train(exp_config)
