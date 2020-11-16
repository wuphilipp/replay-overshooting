import datetime
import pickle
import random
from typing import Callable, List, Optional, Tuple

import numpy as np
import sdeint
from fannypack.utils import pdb_safety_net
from matplotlib.backends.backend_pdf import PdfPages

from dynamics_learning.data.datasets import ContinuousDynamicsDatasetConfig
from dynamics_learning.networks.dynamics import DynamicSystem, Pendulum
from dynamics_learning.utils.data_utils import Sampler, UniformSampler
from dynamics_learning.utils.plot_utils import PlotHandler as ph

pdb_safety_net()

datetime_str = f"{datetime.datetime.now().strftime('%a-%H-%M-%S')}"

dt = 0.025 * 2
trajectory_length = 101
train_end_time = (trajectory_length - 1) * dt


def policy_generator(index: int) -> Callable[[np.ndarray, float], np.ndarray]:
    """Generates a policy.

    Parameters
    ----------
    index : int
        Input dataset index.

    Returns
    -------
    Callable[[np.ndarray, float], np.ndarray]
        Returns a policy.
    """
    points = 20
    if index % 3 == 0:
        random_values = np.random.uniform(low=-10, high=10, size=points)
    if index % 3 == 1:
        random_values = np.random.uniform(low=-0.5, high=0.5, size=points)
    if index % 3 == 2:
        random_values = np.random.uniform(low=-4, high=4, size=points)

    sampled_list = random.sample(list(range(points)), points // 2)
    random_values[sampled_list] = 0

    def policy(x: np.ndarray, t: float) -> np.ndarray:
        """Some time only dependent policy."""
        a = trajectory_length // points
        return np.array([random_values[min(int(t // (dt * a)), points - 1)]])

    return policy


def validation_policy_generator(
    index: int,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Generates a policy.

    Parameters
    ----------
    index : int
        Input dataset index.

    Returns
    -------
    Callable[[np.ndarray, float], np.ndarray]
        Returns a policy.
    """
    return policy_generator(index)
    points = 10
    if index % 3 == 0:
        random_values = np.random.uniform(low=-10, high=10, size=10)
    if index % 3 == 1:
        random_values = np.random.uniform(low=-0.5, high=0.5, size=10)
    if index % 3 == 2:
        random_values = np.random.uniform(low=-4, high=4, size=10)

    sampled_list = random.sample(list(range(10)), 5)
    random_values[sampled_list] = 0

    def policy(x: np.ndarray, t: float) -> np.ndarray:
        """Some time only dependent policy."""
        return np.array([random_values[min(int(t // (dt * points)), points - 1)]])

    return policy_generator(index)


class DataSetGenerator:
    """Generator for creating a datset."""

    def __init__(
        self,
        system: DynamicSystem,
        data_config: ContinuousDynamicsDatasetConfig,
        policy_generator: Callable[[int], Callable[[np.ndarray, float], np.ndarray]],
        generation_batches: int = 100,
        process_noise: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize a dataset generator.

        Parameters
        ----------
        system : DynamicSystem
            Dynamic system.
        data_config : ContinuousDynamicsDatasetConfig
            The configuration of the data.
        policy_generator : Callable[[int], Callable[[np.ndarray, float], np.ndarray]]
            A policy generator.
        generation_batches : int
            Number of data points to generate at a time.
        process_noise : Optional[np.ndarray], default=None
            Noise to use in the SDE.
        """
        self._length = data_config.num_trajectories
        self._sys = system
        self._generation_batches = generation_batches
        self._process_noise = process_noise
        self._policy_generator = policy_generator
        self.data = self.create_data(data_config)

    def create_data(
        self, config: ContinuousDynamicsDatasetConfig
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate a dataset from a configuration.

        Parameters
        ----------
        config : ContinuousDynamicsDatasetConfig
            The configuration of the data.

        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
            A list of [time, state, control] trajectories.
        """
        data = []
        remaining = config.num_trajectories

        # stochastic
        if self._process_noise is not None:
            iterations = 0
            for i in range(remaining):
                tspan = np.linspace(
                    0.0, config.end_times.sample(), config.trajectory_length
                )
                x0 = config.ic.sample()

                # noise
                def G(x, t):
                    return self._process_noise

                policy = self._policy_generator(i)

                # dynamics
                def dyn(x, t):
                    return self._sys.dx(x, t, p=None, u=policy(x, t))

                while True:
                    # sample dynamics parameters
                    result = sdeint.itoint(dyn, G, x0, tspan)  # type: ignore
                    if not np.isnan(np.sum(result)):
                        # sdeint gets nan sometimes?
                        break
                    if iterations > self._generation_batches and iterations % 1000 == 0:
                        print(f"remaining: {i}")
                    iterations += 1

                # wrapping angles to [-pi, pi]
                # > https://stackoverflow.com/a/11181951

                # if isinstance(self._sys, Pendulum):
                #     result[:, 0] = np.arctan2(
                #         np.sin(result[:, 0]), np.cos(result[:, 0])
                #     )

                # reconstructing (deterministic) control inputs
                _ctrl: List[np.ndarray] = []
                for i in range(len(tspan)):
                    t = tspan[i]
                    x = result[i, :]
                    _ctrl.append(policy(x, t))
                ctrl: np.ndarray = np.array(_ctrl)
                data.append((tspan, result, ctrl))
            return data
        return data

    def plot_generated_data(self) -> None:
        """Visualize the generated data"""
        for t, state, ctrl in self.data:
            with ph.plot_context(sp_shape=(3, 1)) as (fig, axs):
                axs[0].plot(t, state[:, 0])
                axs[1].plot(t, state[:, 1])
                axs[2].plot(t, ctrl)
                ph.show()

    def save_all_data(self, name, max_count=100) -> None:
        with PdfPages(f"{name}-{datetime_str}.pdf") as pdf:
            for i, (t, state, ctrl) in enumerate(self.data):
                if i == max_count:
                    break
                with ph.plot_context(sp_shape=(3, 1)) as (fig, axs):
                    axs[0].set_title(f"data point {i}")
                    axs[0].set_ylabel("theta")
                    axs[0].plot(t, state[:, 0])
                    axs[1].plot(t, state[:, 1])
                    axs[1].set_ylabel("theta dot")
                    axs[2].plot(t, ctrl)
                    axs[2].set_xlabel("time")
                    axs[2].set_ylabel("torque")

                    axs[0].set_ylim((-np.pi, np.pi))
                    axs[1].set_ylim((-4, 4))
                    axs[2].set_ylim((-11, 11))

                    for ax in axs:
                        ax.plot(t, np.zeros_like(t), "--")

                    pdf.savefig(fig)


np.random.seed(0)

data_config = ContinuousDynamicsDatasetConfig(
    ic=UniformSampler(np.array([-np.pi, -0.5]), np.array([np.pi, 0.5])),
    end_times=UniformSampler(train_end_time, train_end_time),
    trajectory_length=trajectory_length,
    num_trajectories=10000,
)
process_noise = np.array([[0.025, 0], [0, 0.025]])
dynamics = Pendulum(friction=0.5)

train_data = DataSetGenerator(
    dynamics, data_config, policy_generator, process_noise=process_noise
)

data_config = ContinuousDynamicsDatasetConfig(
    ic=UniformSampler(np.array([-np.pi, -1]), np.array([np.pi, 1])),
    end_times=UniformSampler(train_end_time, train_end_time),
    trajectory_length=trajectory_length,
    num_trajectories=32,
)
val_data = DataSetGenerator(
    dynamics, data_config, validation_policy_generator, process_noise=process_noise
)

with open(f"pend_train-{datetime_str}.pkl", "wb") as handle:
    pickle.dump(train_data.data, handle)
with open(f"pend_val-{datetime_str}.pkl", "wb") as handle:
    pickle.dump(val_data.data, handle)

val_data.save_all_data("val_data")
train_data.save_all_data("train_data")


# Test curated data for visualization


class DeterministicSampler(Sampler):
    def __init__(self):
        self._counter = -1

    def sample(self) -> np.ndarray:
        """Returns a single sample."""
        self._counter += 1
        if self._counter in [0, 1, 2, 3]:
            return np.array([0.0, 0.0])
        if self._counter in [4, 5, 6, 7]:
            return np.array([1.0, 0.0])
        if self._counter in [8, 9, 10, 11]:
            return np.array([-1.0, 0.0])

    def sample_batch(self, batch_size: int) -> np.ndarray:
        """Returns a batch of samples."""
        raise NotImplementedError


def test_policy_generator(index: int,) -> Callable[[np.ndarray, float], np.ndarray]:
    """Generates a policy.

    Parameters
    ----------
    index : int
        Input dataset index.

    Returns
    -------
    Callable[[np.ndarray, float], np.ndarray]
        Returns a policy.
    """
    if index % 4 == 0:
        return lambda x, t: np.array([0])
    elif index % 4 == 1:
        return lambda x, t: 5 * np.sin([t * 2])
    elif index % 4 == 2:
        return lambda x, t: np.array([5])
    else:
        return lambda x, t: np.array([-5])


data_config = ContinuousDynamicsDatasetConfig(
    ic=DeterministicSampler(),
    end_times=UniformSampler(train_end_time, train_end_time),
    trajectory_length=trajectory_length,
    num_trajectories=12,
)
test_data = DataSetGenerator(
    dynamics, data_config, test_policy_generator, process_noise=process_noise
)

with open(f"pend_test-{datetime_str}.pkl", "wb") as handle:
    pickle.dump(test_data.data, handle)

test_data.save_all_data("test_data")
