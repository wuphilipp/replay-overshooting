import os
import tempfile
import time
from dataclasses import replace
from typing import Optional

import numpy as np
import torch
import yaml
from fannypack.utils import Buddy, get_git_commit_hash, pdb_safety_net
from torch import nn as nn

from dynamics_learning.data.datasets import DatasetConfig, MITPushDatasetConfig
from dynamics_learning.training.configs import ExpConfig


def count_parameters(model: nn.Module, trainable: bool = False) -> int:
    """Helper to count the number of parameters in a model.

    Parameters
    ----------
    model : nn.Module
        The model of interest.
    trainable : bool
        Whether to count the number of trainable parameters.

    Returns
    -------
    int
        Number of parameters
    """
    if trainable:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_valid(exp_config: ExpConfig) -> None:
    """Check if model evaluation is possible.

    Parameters
    ----------
    exp_config : ExpConfig
        Specifies all information regarding the experiment.
    """
    if get_git_commit_hash() != exp_config.git_commit_hash:
        print("[Warning] - the current commit does not match that of the model.")
        print(
            f"Current commit: {get_git_commit_hash()}. Model commit: {exp_config.git_commit_hash}"
        )
    elif exp_config.name is None:
        raise ValueError(f"exp_config.name must not be {None}")


def load_metadata(experiment_name: str) -> ExpConfig:
    """Load an `ExpConfig` from a file."""
    # HACK for loading in metadata with unsafe mode
    # TODO serialise ExperimentConfig properly
    try:
        metadata_dir = "metadata"
        path = os.path.join(metadata_dir, f"{experiment_name}.yaml")
        with open(path, "r") as file:
            metadata = yaml.load(file, Loader=yaml.Loader)
            print("Loaded metadata:", metadata)
    except Exception:
        # Does not exist so look into local file dir
        path = f"{experiment_name}.yaml"
        with open(path, "r") as file:
            metadata = yaml.load(file, Loader=yaml.Loader)
            print("Loaded metadata:", metadata)
    return metadata


@torch.no_grad()
def evaluate(
    exp_config: ExpConfig,
    dataset_config: DatasetConfig,
    experiment_name: Optional[str] = None,
    save_summary: bool = False,
    debug: bool = False,
) -> None:
    """Evaluates a network.

    Parameters
    ----------
    exp_config : ExpConfig
        Specifies all information regarding the experiment.
    experiment_name : str
        Name of experiment to load.
    save_summary : bool, default=False
        Flag indicating whether to visualize a summary of the evaluation.
    debug : bool, default=False
        Flag indicating whether to evaluate in debug mode.
    """
    pdb_safety_net()

    # set random seed for repeatability
    np.random.seed(0)
    torch.manual_seed(0)
    check_valid(exp_config)

    # build model
    estimator = exp_config.model.create()

    # set up buddy
    # some hacks to get around serialization
    # TODO figure out a better way to do this
    metadata_dir = "metadata"
    dir = tempfile.gettempdir()
    assert exp_config.name is not None
    buddy = Buddy(exp_config.name, estimator, optimizer_type="adam", metadata_dir=dir)
    buddy._metadata_dir = metadata_dir

    # load the checkpoint
    if experiment_name is None:
        buddy.load_checkpoint()
    else:
        buddy.load_checkpoint(label="final", experiment_name=experiment_name)

    # provide network diagnostics.
    print()
    print("Model Architecture:")
    print(estimator)
    print(f"Total parameters: {count_parameters(estimator)}")
    print(f"Total trainable parameters: {count_parameters(estimator, trainable=True)}")
    print(f"Latent dim size: {exp_config.model.latent_dim}")
    print()

    # model performance
    if debug:
        dataset_config = replace(dataset_config, num_viz_trajectories=3)
    dataset = dataset_config.create()
    vis_data = dataset.get_viz_data(buddy.device)

    # reporting eval losses: (vis_data, filter_times, predict_times)
    start_time = time.time()

    if isinstance(dataset_config, MITPushDatasetConfig):
        prediction_points = 25
    else:
        prediction_points = 50

    estimator.eval_loss(vis_data, 5, prediction_points)
    estimator.eval_loss(vis_data, 25, prediction_points)
    end_time = time.time()
    print(f"Total evaluation time: {end_time - start_time}")

    # summary plots
    if save_summary:
        assert exp_config.name is not None
        estimator.summary_plot(vis_data, exp_config.name, debug=debug)
