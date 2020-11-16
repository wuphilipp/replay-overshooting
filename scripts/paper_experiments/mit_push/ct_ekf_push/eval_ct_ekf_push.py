from dynamics_learning.data.datasets import MITPushDatasetConfig
from dynamics_learning.training.evaluation import evaluate, load_metadata

if __name__ == "__main__":
    dataset_config = MITPushDatasetConfig(
        traj_len=50,
        num_viz_trajectories=20,
        pixel_res=256,
        raw=True,
        cond=False,
        half_image_size=False,
        kloss_dataset=True,  # true for mit push
    )

    name = "ct_ekf_push"
    exp_config = load_metadata(name)
    evaluate(exp_config, dataset_config, experiment_name=name, save_summary=True)
