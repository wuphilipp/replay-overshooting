from collections import namedtuple

from fannypack.utils import pdb_safety_net

from dynamics_learning.custom.lr_functions import lr5
from dynamics_learning.data.datasets import MITPushDatasetConfig
from dynamics_learning.networks.baseline.ssm_baseline import GSSMBaselineConfig
from dynamics_learning.networks.estimator import EstimatorConfig
from dynamics_learning.training.configs import ExpConfig
from dynamics_learning.training.experiments import train

model_config: EstimatorConfig
exp_config: ExpConfig
pdb_safety_net()


hyperparameter_defaults = dict(
    batch_size=8, learning_rate=1e-3, epochs=400, latent_dim=16
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

model_config = GSSMBaselineConfig(
    latent_dim=hy_config.latent_dim,
    latent_obs_dim=16,
    hidden_units=64,
    ctrl_dim=6,
    dataset=dataset_config,
)

exp_config = ExpConfig(
    name="gssm_push",
    model=model_config,
    ramp_iters=200,
    batch_size=hy_config.batch_size,
    epochs=hy_config.epochs,
    log_iterations_simple=10,
    log_iterations_images=100,
    base_learning_rate=hy_config.learning_rate,
    learning_rate_function=lr5,
    gradient_clip_max_norm=500,
)
train(exp_config)
