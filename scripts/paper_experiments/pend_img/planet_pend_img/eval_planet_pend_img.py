from dynamics_learning.data.datasets import (
    ImageDynamicDatasetConfig,
    PendFixedDatasetConfig,
)
from dynamics_learning.training.evaluation import evaluate, load_metadata

if __name__ == "__main__":
    pend_fixed = PendFixedDatasetConfig(traj_len=33, num_viz_trajectories=1000)
    dataset_config = ImageDynamicDatasetConfig(
        traj_len=pend_fixed.traj_len,
        num_trajectories=10000,
        num_viz_trajectories=pend_fixed.num_viz_trajectories,
        system=pend_fixed,
        policy=None,
    )

    name = "planet_pend_img"
    exp_config = load_metadata(name)

    evaluate(exp_config, dataset_config, experiment_name=name, save_summary=True)
