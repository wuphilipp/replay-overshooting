import itertools
import pickle
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import fannypack
import numpy as np
import sdeint
import torch
from torch.utils.data import Dataset

from dynamics_learning.networks.dynamics import DynamicSystem, Pendulum, VanDerPol
from dynamics_learning.utils.data_utils import (
    GaussianSampler,
    Sampler,
    UniformSampler,
    prep_batch,
)
from dynamics_learning.utils.plot_utils import Axis, PlotSettings

# ------------------------- #
# VISUALIZATION DATACLASSES #
# ------------------------- #


@dataclass(frozen=True)
class TrajectoryNumpy:
    """Holds the information about a trajectory."""

    states: np.ndarray
    observations: Dict[str, np.ndarray]
    controls: np.ndarray


@dataclass(frozen=True)
class VisData:
    """Struct to hold data for plotting."""

    t: torch.Tensor
    y0: torch.Tensor
    y: torch.Tensor
    u: torch.Tensor
    np_t: np.ndarray
    np_y: np.ndarray
    np_u: np.ndarray
    plot_settings: PlotSettings


@dataclass(frozen=True)
class VisDataIMG(VisData):
    """Struct to hold data for plotting image data."""

    pv: torch.Tensor
    np_pv: np.ndarray


# ------- #
# HELPERS #
# ------- #


def _get_viz_data_basic(dataset: "DynamicsDataset", device: torch.device) -> VisData:
    """Helper for returning only times and associated data, no extra info."""
    assert hasattr(dataset, "_viz_data")
    assert callable(getattr(dataset, "get_default_plot_settings", None))

    t_list = [t for t, x, u in dataset._viz_data]  # type: ignore
    x_list = [x for t, x, u in dataset._viz_data]  # type: ignore
    u_list = [u for t, x, u in dataset._viz_data]  # type: ignore

    t = torch.tensor(t_list, dtype=torch.float, device=device)[0, :]
    y = torch.tensor(x_list, dtype=torch.float, device=device).transpose(0, 1)
    u = torch.tensor(u_list, dtype=torch.float, device=device).transpose(0, 1)

    return VisData(
        t=t,
        y0=y[0],
        y=y,
        u=u,
        np_t=t.clone().cpu().numpy(),
        np_y=y.clone().cpu().numpy(),
        np_u=u.clone().cpu().numpy(),
        plot_settings=dataset.get_default_plot_settings(),
    )


# -------- #
# DATASETS #
# -------- #


@dataclass
class DatasetConfig:
    """Configuration for a Dataset for training."""

    traj_len: int
    num_viz_trajectories: int

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Observation dimension."""
        raise NotImplementedError

    def create(self) -> "DynamicsDataset":
        """Create a `DynamicsDataset`."""
        raise NotImplementedError


@dataclass
class DummyDatasetConfig(DatasetConfig):
    """A dummy dataset configuration for nested definitions."""

    traj_len: int = 0
    num_viz_trajectories: int = 0


class DynamicsDataset(ABC, Dataset):
    """Abstract dataset interface for dynamic systems."""

    # TODO fix typing
    _data: Any
    # _data: List[Tuple[np.ndarray, np.ndarray]]
    _viz_data: Any
    # _viz_data: List[Tuple[np.ndarray, np.ndarray]]

    def __len__(self) -> int:
        """Length of dataset."""
        return len(self._data)

    def __getitem__(self, idx: Union[int, torch.Tensor, List[int]]) -> torch.Tensor:
        """Get specific datapoint."""
        if torch.is_tensor(idx):
            assert isinstance(idx, torch.Tensor)  # mypy
            idx = idx.tolist()
        # TODO fix
        return self._data[idx]  # type: ignore

    @abstractmethod
    def get_viz_data(self, device: torch.device) -> VisData:
        """Get a VisData object of the viz dataset."""

    @abstractmethod
    def get_default_plot_settings(self) -> PlotSettings:
        """Get plot settings for each system."""

    def preprocess_data(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare a batch for training."""
        _, batch_t, batch_y, batch_u = prep_batch(batch, device)
        return batch_t, batch_y, batch_u


@dataclass(frozen=True)
class ContinuousDynamicsDatasetConfig:
    """Config object for continuous dynamical systems."""

    ic: Sampler
    end_times: Sampler
    policy: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    trajectory_length: int = 20
    num_trajectories: int = 10000


class ContinuousDynamicsDataset(DynamicsDataset):
    """Dataset for DE-defined continuous dynamical systems."""

    def __init__(
        self,
        system: DynamicSystem,
        data_config: ContinuousDynamicsDatasetConfig,
        viz_config: ContinuousDynamicsDatasetConfig,
        generation_batches: int = 100,
        process_noise: Optional[np.ndarray] = None,
        measurement_var: float = 0.0,
        param_sampler: Optional[Sampler] = None,
    ) -> None:
        """Initalize a continuous system.

        Parameters
        ----------
        system : DynamicSystem
            Dynamic system.
        data_config : ContinuousDynamicsDatasetConfig
            The configuration of the data.
        viz_config : ContinuousDynamicsDatasetConfig
            The configuration of the visualization data.
        generation_batches : int
            Number of data points to generate at a time.
        process_noise : Optional[np.ndarray], default=None
            Noise to use in the SDE.
            # THIS ONLY WORKS FOR Van Der Pol and pendulum right now.
        measurement_var : float, default=0.0
            Measurement variance to be used during pre processing. By default, there is no added noise.
        """
        self._length = data_config.num_trajectories
        self._sys = system
        self._generation_batches = generation_batches
        self._process_noise = process_noise
        self._measurement_var = measurement_var
        self._param_sampler = param_sampler
        self._data = self.create_data(data_config)
        self._viz_data = self.create_data(viz_config)

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
        data : List[Tuple[np.ndarray, np.ndarray]], shape=[(T, (T, B, X))]
            List of tuples of times and batches of sequences of data.
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

                # dynamics
                if config.policy is None:

                    def dyn(x, t):
                        return self._sys.dx(x, t, p=p, u=None)

                else:

                    def dyn(x, t):
                        return self._sys.dx(x, t, p=p, u=config.policy(x, t))

                while True:
                    # sample dynamics parameters
                    if self._param_sampler is not None:
                        p = self._param_sampler.sample()
                    else:
                        p = None

                    result = sdeint.itoint(dyn, G, x0, tspan)  # type: ignore
                    if not np.isnan(np.sum(result)):
                        # sdeint gets nan sometimes?
                        break
                    if iterations > self._generation_batches and iterations % 1000 == 0:
                        print(f"remaining: {i}")
                    iterations += 1

                # wrapping angles to [-pi, pi]
                # > https://stackoverflow.com/a/11181951
                if isinstance(self._sys, Pendulum):
                    result[:, 0] = np.arctan2(
                        np.sin(result[:, 0]), np.cos(result[:, 0])
                    )

                # reconstructing (deterministic) control inputs
                _ctrl: List[np.ndarray] = []
                for i in range(len(tspan)):
                    t = tspan[i]
                    x = result[i, :]
                    if config.policy is None:
                        _ctrl.append([])
                    else:
                        _ctrl.append(config.policy(x, t))
                ctrl: np.ndarray = np.array(_ctrl)
                data.append((tspan, result, ctrl))
            return data

        # deterministic
        while remaining > 0:
            raise NotImplementedError  # TODO: add back support for this later

            batch_size = min(remaining, self._generation_batches)
            t = np.linspace(0.0, config.end_times.sample(), config.trajectory_length)
            x0s = torch.tensor(config.ic.sample_batch(batch_size), dtype=torch.float)
            xs = self._sys.solve_torch(t, x0s)

            # wrapping angles to [-pi, pi]
            # > https://stackoverflow.com/a/11181951
            if isinstance(self._sys, Pendulum):
                xs[..., 0] = np.arctan2(np.sin(xs[..., 0]), np.cos(xs[..., 0]))

            data += [(t, xs[:, i, :]) for i in range(batch_size)]
            remaining = remaining - batch_size
        return data

    def get_viz_data(self, device: torch.device) -> VisData:
        """Get a VisData object of the viz dataset."""
        return _get_viz_data_basic(self, device)

    def get_default_plot_settings(self) -> PlotSettings:
        """Get plot settings for each system."""
        if isinstance(self._sys, Pendulum):
            # angular plot settings
            return PlotSettings(
                axis=Axis(xlim=(-np.pi - 0.1, np.pi + 0.1), ylim=(-7, 7),)
            )
        return PlotSettings(axis=Axis(xlim=(-5, 5), ylim=(-5, 5)))

    def preprocess_data(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """See parent."""
        _, batch_t, batch_y, batch_u = prep_batch(batch, device)
        return batch_t, batch_y, batch_u


@dataclass
class OfflineDatasetConfig(DatasetConfig):
    """Dataset for training that reads from a saved pickle file."""

    paths: Tuple[str, str]
    pend_xy: bool = False

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Observation dimension."""
        return (2,)

    def create(self) -> "DynamicsDataset":
        """Create a `DynamicsDataset`."""
        dataset = OfflineDataset(
            self.traj_len, self.num_viz_trajectories, self.paths[0], self.paths[1],
        )
        if self.pend_xy:
            dataset = PendulumXYDataset(dataset, has_vel=False)
        return dataset


@dataclass
class PendFixedDatasetConfig(DatasetConfig):
    """Configuration for saved generated pendulum dataset."""

    paths: Tuple[str, str] = (
        "https://drive.google.com/file/d/1KFcM0e5fhCzUbVPxaUj0_bOp0RS_3Myw/view?usp=sharing",
        "https://drive.google.com/file/d/1HTS6_v55mFw6tFv9VEigdDTMB9n92UEb/view?usp=sharing",
    )
    pend_xy: bool = False

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Observation dimension."""
        return (2,)

    def create(self) -> "DynamicsDataset":
        """Create a `DynamicsDataset`."""
        path_train = fannypack.data.cached_drive_file(
            "pend_train" + self.paths[0], self.paths[0]
        )
        path_val = fannypack.data.cached_drive_file(
            "pend_val" + self.paths[1], self.paths[1]
        )
        dataset = OfflineDataset(
            self.traj_len, self.num_viz_trajectories, path_train, path_val,
        )
        if self.pend_xy:
            dataset = PendulumXYDataset(dataset, has_vel=False)
        return dataset


class OfflineDataset(DynamicsDataset):
    """Dataset for DE-defined continuous dynamical systems."""

    def __init__(
        self,
        traj_len: int,
        num_viz_trajectories: int,
        train_data_path: str,
        val_data_path: str,
    ) -> None:
        """Initalize an offline saved dataset."""
        with open(train_data_path, "rb") as handle:
            self._data = pickle.load(handle)
        assert len(self._data[0][0]) >= traj_len
        self._data = [
            (t[:traj_len], x[:traj_len], c[:traj_len]) for (t, x, c) in self._data
        ]
        with open(val_data_path, "rb") as handle:
            self._viz_data = pickle.load(handle)[:num_viz_trajectories]

    def get_viz_data(self, device: torch.device) -> VisData:
        """Get a VisData object of the viz dataset."""
        return _get_viz_data_basic(self, device)

    def get_default_plot_settings(self) -> PlotSettings:
        """Get plot settings for each system."""
        return PlotSettings(axis=Axis(xlim=(-np.pi - 0.1, np.pi + 0.1), ylim=(-7, 7),))

    def preprocess_data(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """See parent."""
        _, batch_t, batch_y, batch_u = prep_batch(batch, device)
        return batch_t, batch_y, batch_u


class PendulumXYDataset(ContinuousDynamicsDataset):
    """Dataset for XY (and possible velocity) pend data."""

    def __init__(
        self,
        theta_dataset: ContinuousDynamicsDataset,
        has_vel: bool = False,
        offset: bool = False,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        theta_dataset : ContinuousDynamicsDataset
            A dataset of XY pendulum data.
        has_vel : bool, default=False
            Flag indicating whether velocity data is returned.
        offset : bool, default=False
            Flag indicating whether there are random planar offsets in pendula centers.
        """
        # getting training and viz angular data
        train_data = np.stack([pt[1] for pt in theta_dataset._data], axis=1)
        viz_data = np.stack([pt[1] for pt in theta_dataset._viz_data], axis=1)

        # converting to XY data
        train_xy = np.zeros_like(train_data)
        train_xy[..., 0] = np.sin(train_data[..., 0])
        train_xy[..., 1] = -np.cos(train_data[..., 0])
        viz_xy = np.zeros_like(viz_data)
        viz_xy[..., 0] = np.sin(viz_data[..., 0])
        viz_xy[..., 1] = -np.cos(viz_data[..., 0])

        # generate offset
        if offset:
            train_offset = np.random.randn(2)
            viz_offset = np.random.randn(2)
        else:
            train_offset = np.zeros(2)
            viz_offset = np.zeros(2)

        # generate velocity data
        if has_vel:
            train_vel = np.zeros_like(train_data)
            train_vel[..., 0] = train_data[..., 1] * np.cos(train_data[..., 0])
            train_vel[..., 1] = train_data[..., 1] * np.sin(train_data[..., 0])
            viz_vel = np.zeros_like(viz_data)
            viz_vel[..., 0] = viz_data[..., 1] * np.cos(viz_data[..., 0])
            viz_vel[..., 1] = viz_data[..., 1] * np.sin(viz_data[..., 0])

            train_data = np.concatenate((train_xy + train_offset, train_vel), axis=-1)
            viz_data = np.concatenate((viz_xy + viz_offset, viz_vel), axis=-1)
        else:
            train_data = train_xy + train_offset
            viz_data = viz_xy + viz_offset

        # packing training and viz data
        self._data = [
            (theta_dataset._data[i][0], train_data[:, i, :], theta_dataset._data[i][2])
            for i in range(train_data.shape[1])
        ]
        self._viz_data = [
            (
                theta_dataset._viz_data[i][0],
                viz_data[:, i, :],
                theta_dataset._viz_data[i][2],
            )
            for i in range(viz_data.shape[1])
        ]

    def get_viz_data(self, device: torch.device) -> VisData:
        """Get a VisData object of the viz dataset."""
        return _get_viz_data_basic(self, device)

    def get_default_plot_settings(self) -> PlotSettings:
        """Get plot settings for each system."""
        return PlotSettings(axis=Axis(xlim=(-3, 3), ylim=(-3, 3),))


class PendulumImageDataset(DynamicsDataset):
    """Dataset for pendulum images."""

    def __init__(
        self, xy_dataset: PendulumXYDataset, img_size: int = 16, pixel_res: int = 256,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        xy_dataset : PendulumXYDataset
            A dataset consisting of XY pendulum data to be converted to images.
        img_size : int, default=16
            The side length of the (square) images to be generated.
        pixel_res : int, default=256
            Color channel resolution for each pixel.
        """
        self._img_size = img_size
        self._pixel_res = pixel_res

        # expand_dims to add the single color channel to the data
        train_data = np.expand_dims(
            self.pend_xy_to_image(
                torch.tensor(
                    np.stack([pt[1][..., 0:2] for pt in xy_dataset._data], axis=1),
                )
            ),
            axis=-3,
        )
        viz_data = np.expand_dims(
            self.pend_xy_to_image(
                torch.tensor(
                    np.stack([pt[1][..., 0:2] for pt in xy_dataset._viz_data], axis=1),
                )
            ),
            axis=-3,
        )
        self._data = [
            (
                xy_dataset._data[i][0],  # time
                train_data[:, i, ...],  # img data
                xy_dataset._data[i][2],  # ctrl inputs
                xy_dataset._data[i][1],  # original xy data
            )
            for i in range(train_data.shape[1])
        ]
        self._viz_data = [
            (
                xy_dataset._viz_data[i][0],
                viz_data[:, i, ...],
                xy_dataset._viz_data[i][2],
                xy_dataset._viz_data[i][1],
            )
            for i in range(viz_data.shape[1])
        ]

    def pend_xy_to_image(
        self,
        xy_data: torch.Tensor,
        r: float = 0.6,
        pos_max: float = 2.0,
        pos_min: float = -2.0,
    ) -> np.ndarray:
        """Generates images from sequences of batched xy data.

        Parameters
        ----------
        xy_data : torch.Tensor, shape=(T, B, 2)
            Batched sequences of pendulum mass xy coordinates.
        r : float, default=0.5
            Radius of the (assumed circuluar) pendulum mass.
        size : int, default=16
            Side length of the (square) image.
        pos_max : float, default=2.0
            Largest value of view frame coordinates.
        pos_min : float, default=-2.0
            Smallest value of view frame coordinates.

        Returns
        -------
        img : np.ndarray, shape=(T, B, size, size)
            Image data encoding the data.
        """
        # shape
        assert len(xy_data.shape) == 3
        T, B = xy_data.shape[:2]
        x = -xy_data[..., 1:2]  # swap x and y in image space
        y = xy_data[..., 0:1]

        # initialize img
        img = np.ones((T, B, self._img_size, self._img_size))  # white background

        # convert xy and r to pixel space
        assert (x >= pos_min).all() and (x <= pos_max).all()
        assert (y >= pos_min).all() and (y <= pos_max).all()
        gap = pos_max - pos_min
        x_px = ((self._img_size - 1) / gap) * (x.numpy() - pos_min)
        y_px = ((self._img_size - 1) / gap) * (y.numpy() - pos_min)
        r_px = ((self._img_size - 1) / gap) * r  # float

        # getting all pixel index pairs in image
        _vals = [i for i in range(self._img_size)]
        pairs = np.array(
            [list(pair) for pair in itertools.product(_vals, _vals)]
        ).astype(float)[:, None, None, :]
        pairs = np.repeat(pairs, T, axis=1)
        pairs = np.repeat(pairs, B, axis=2)

        # computing mass indices for each time step
        dists = np.sqrt(
            (pairs[..., 0:1] - x_px) ** 2.0 + (pairs[..., 1:2] - y_px) ** 2.0
        ).squeeze(-1)
        inds = dists <= r_px + 0.5

        # generating pixel data
        for i in range(T):
            for j in range(B):
                mass_pairs = pairs[:, i, j, :][inds[:, i, j]].astype(int)
                mass_dists = dists[:, i, j][inds[:, i, j]][..., np.newaxis]
                img[i, j, mass_pairs[..., 0:1], mass_pairs[..., 1:2]] = mass_dists / (
                    r_px + 0.5
                )
        img = np.maximum(
            img - 0.05 * np.random.rand(T, B, self._img_size, self._img_size), 0.0,
        )  # noise
        img = np.round(img * (self._pixel_res - 1)) / (self._pixel_res - 1)  # quantize

        return img  # values between 0 and 1

    def get_viz_data(self, device: torch.device) -> VisData:
        """Get a VisData object of the viz dataset."""
        t_list = [t for t, x, u, pv in self._viz_data]
        x_list = [x for t, x, u, pv in self._viz_data]
        u_list = [u for t, x, u, pv in self._viz_data]
        pv_list = [pv for t, x, u, pv in self._viz_data]

        t = torch.tensor(t_list, dtype=torch.float, device=device)[0, :]
        y = torch.tensor(x_list, dtype=torch.float, device=device).transpose(0, 1)
        u = torch.tensor(u_list, dtype=torch.float, device=device).transpose(0, 1)
        pv = torch.tensor(pv_list, dtype=torch.float, device=device).transpose(0, 1)

        return VisDataIMG(
            t=t,
            y0=y[0],
            y=y,
            u=u,
            np_t=t.clone().cpu().numpy(),
            np_y=y.clone().cpu().numpy(),
            np_u=u.clone().cpu().numpy(),
            plot_settings=self.get_default_plot_settings(),
            pv=pv,
            np_pv=pv.clone().cpu().numpy(),
        )

    def get_default_plot_settings(self) -> PlotSettings:
        """Get plot settings for each system."""
        return PlotSettings(
            axis=Axis(xlim=(0, self._img_size - 1), ylim=(0, self._img_size - 1))
        )


class DynamicsDatasetType(Enum):
    """Supported simple dynamics datasets."""

    VDP = 1
    PEND_XY = 2
    PEND_ALL = 3
    PEND_ANG = 4


@dataclass
class DynamicSystemDatasetConfig(DatasetConfig):
    """Supported synthetic datasets."""

    system: DynamicsDatasetType
    num_trajectories: int = 10000
    dt: float = 0.025
    measurement_var: float = 0.0001
    policy: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Observation dimension."""
        return (2,) if self.system != DynamicsDatasetType.PEND_ALL else (4,)

    def create(self) -> "DynamicsDataset":
        """Create a `DynamicsDataset`."""
        train_end_time = (self.traj_len - 1) * self.dt
        valid_end_time = 3 * (self.traj_len - 1) * self.dt
        process_noise = np.array([[0.05, 0], [0, 0.05]])

        if self.system == DynamicsDatasetType.VDP:
            vdp_sys = VanDerPol(mu=1)
            data_config = ContinuousDynamicsDatasetConfig(
                ic=GaussianSampler(np.array([0, 0]), np.array([[4, 0], [0, 4]])),
                end_times=UniformSampler(train_end_time, train_end_time),
                policy=self.policy,
                trajectory_length=self.traj_len,
                num_trajectories=self.num_trajectories,
            )
            viz_config = ContinuousDynamicsDatasetConfig(
                ic=GaussianSampler(np.array([0, 0]), np.array([[4, 0], [0, 4]])),
                end_times=UniformSampler(valid_end_time, valid_end_time),
                policy=self.policy,
                trajectory_length=3 * (self.traj_len - 1) + 1,
                num_trajectories=self.num_viz_trajectories,
            )
            dataset = ContinuousDynamicsDataset(
                vdp_sys,
                data_config,
                viz_config,
                process_noise=process_noise,
                measurement_var=self.measurement_var,
            )

        elif (
            self.system == DynamicsDatasetType.PEND_ANG
            or self.system == DynamicsDatasetType.PEND_XY
            or self.system == DynamicsDatasetType.PEND_ALL
        ):
            p_sys = Pendulum()

            # angular dataset
            data_config = ContinuousDynamicsDatasetConfig(
                ic=UniformSampler(np.array([-np.pi, -2]), np.array([np.pi, 2])),
                end_times=UniformSampler(train_end_time, train_end_time),
                policy=self.policy,
                trajectory_length=self.traj_len,
                num_trajectories=self.num_trajectories,
            )
            viz_config = ContinuousDynamicsDatasetConfig(
                ic=UniformSampler(np.array([-np.pi, -2]), np.array([np.pi, 2])),
                end_times=UniformSampler(valid_end_time, valid_end_time),
                policy=self.policy,
                trajectory_length=3 * (self.traj_len - 1) + 1,
                num_trajectories=self.num_viz_trajectories,
            )
            dataset = ContinuousDynamicsDataset(
                p_sys,
                data_config,
                viz_config,
                process_noise=process_noise,
                measurement_var=self.measurement_var,
            )

            # XY or image dataset
            if (
                self.system == DynamicsDatasetType.PEND_XY
                or self.system == DynamicsDatasetType.PEND_ALL
            ):
                dataset = PendulumXYDataset(
                    dataset, has_vel=self.system == DynamicsDatasetType.PEND_ALL,
                )
        return dataset


class ImageDynamicsDatasetType(Enum):
    """Supported image dynamics datasets."""

    PEND_IMG = 1
    PEND_IMG_ZERO_G = 2


@dataclass
class ImageDynamicDatasetConfig(DatasetConfig):
    """Configuration class for creating an image dataset for training."""

    system: Union[ImageDynamicsDatasetType, Tuple[str, str], PendFixedDatasetConfig]
    pixel_res: int = 256
    num_trajectories: int = 10000
    dt: float = 0.025 * 5
    policy: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    img_size: int = 16

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Observation dimension."""
        return (1, self.img_size, self.img_size)

    def create(self) -> "DynamicsDataset":
        """Create a `DynamicsDataset`."""
        if (
            self.system == ImageDynamicsDatasetType.PEND_IMG
            or self.system == ImageDynamicsDatasetType.PEND_IMG_ZERO_G
        ):
            train_end_time = (self.traj_len - 1) * self.dt
            valid_end_time = 3 * (self.traj_len - 1) * self.dt
            process_noise = np.array([[0.05, 0], [0, 0.05]])

            if self.system == ImageDynamicsDatasetType.PEND_IMG:
                p_sys = Pendulum()
            elif self.system == ImageDynamicsDatasetType.PEND_IMG_ZERO_G:
                p_sys = Pendulum(gravity=0)
            # angular dataset
            data_config = ContinuousDynamicsDatasetConfig(
                ic=UniformSampler(np.array([-np.pi, -2]), np.array([np.pi, 2])),
                end_times=UniformSampler(train_end_time, train_end_time),
                policy=self.policy,
                trajectory_length=self.traj_len,
                num_trajectories=self.num_trajectories,
            )
            viz_config = ContinuousDynamicsDatasetConfig(
                ic=UniformSampler(np.array([-np.pi, -2]), np.array([np.pi, 2])),
                end_times=UniformSampler(valid_end_time, valid_end_time),
                policy=self.policy,
                trajectory_length=3 * (self.traj_len - 1) + 1,
                num_trajectories=self.num_viz_trajectories,
            )
            dataset = ContinuousDynamicsDataset(
                p_sys, data_config, viz_config, process_noise=process_noise,
            )
            # XY or image dataset
        elif isinstance(self.system, Tuple):
            dataset = OfflineDataset(
                self.traj_len,
                self.num_viz_trajectories,
                self.system[0],
                self.system[1],
            )
        elif isinstance(self.system, PendFixedDatasetConfig):
            assert not self.system.pend_xy
            dataset = self.system.create()
        else:
            raise NotImplementedError(f"system {self.system} is not supported")

        dataset = PendulumXYDataset(dataset, has_vel=True)
        img_dataset = PendulumImageDataset(
            dataset, img_size=self.img_size, pixel_res=self.pixel_res
        )
        return img_dataset


class MITPushDataset(DynamicsDataset):
    """Dataset for the mit push dataset.

    https://mcube.mit.edu/push-dataset/index.html
    """

    def __init__(self, config: "MITPushDatasetConfig") -> None:
        """Initialize dataset.

        Parameters
        ----------
        config : MITPushDatasetConfig
            Push dataset config.
        """
        self._pixel_res = config.pixel_res
        self._config = config
        if config.kloss_dataset:
            dataset_names = [
                "kloss_train0.hdf5",
                "kloss_train1.hdf5",
                "kloss_train2.hdf5",
                "kloss_train3.hdf5",
                "kloss_train4.hdf5",
                "kloss_train5.hdf5",
                "kloss_val.hdf5",
            ]
            self.dt = 0.01
            assert config.traj_len <= 50
            trajectories = _load_trajectories(*dataset_names, kloss_dataset=True)
            trajectories = [
                traj
                for traj in trajectories
                if traj.observations["material"][0] in config.materials
            ]
            # ctrl inputs:
            # > planar gripper pos (dim=2)
            # > gripper force sensors (dim=3)
            # > gripper contact boolean (dim=2, one-hot)
            # > planar gripper contact point (dim=2)
            # > planar gripper contact normal vector (dim=2)
            controls = np.array(
                [
                    np.concatenate(
                        (
                            traj.observations["gripper_pos"][..., [0, 2]],
                            np.roll(
                                traj.observations["gripper_pos"][..., [0, 2]],
                                shift=-1,
                                axis=0,
                            )
                            - traj.observations["gripper_pos"][..., [0, 2]],
                            # traj.observations["gripper_sensors"][..., [0, 1, 2]],
                            np.eye(2)[
                                traj.observations["gripper_sensors"][..., -1].astype(
                                    int
                                )
                            ],
                            # traj.observations["contact_point"][..., [0, 2]],
                            # traj.observations["normal"],
                        ),
                        axis=-1,
                    )
                    for traj in trajectories
                ]
            )
            controls[..., -1, 2:4] = 0  # set terminal EE velocity to 0
        else:
            dataset_names = [
                "gentle_push_10.hdf5",
                "gentle_push_100.hdf5",
                "gentle_push_300.hdf5",
                "gentle_push_1000.hdf5",
            ]
            self.dt = 0.01
            assert config.traj_len <= 30
            trajectories = _load_trajectories(*dataset_names, kloss_dataset=False)
            trajectories = [traj for traj in trajectories]
            controls = np.array(
                [
                    np.concatenate(
                        (
                            traj.controls[..., :6],
                            np.eye(2)[
                                traj.observations["gripper_sensors"][..., -1].astype(
                                    int
                                )
                            ],
                        ),
                        axis=-1,
                    )
                    for traj in trajectories
                ]
            )  # CTRL FOR CYLINDER DATASET

        total_trajectories = len(trajectories)

        states = np.array([traj.states for traj in trajectories])
        images = np.array([traj.observations["image"] for traj in trajectories])
        images = images[:, :, None, :, :]
        if config.half_image_size:
            B, T = images.shape[:2]
            images_torch = torch.tensor(images).reshape(-1, 1, 32, 32)
            downsampled_images = torch.nn.AvgPool2d(2, stride=2)(images_torch)
            images = downsampled_images.reshape(B, T, 1, 16, 16).numpy()

        images = np.round(images * (self._pixel_res - 1)) / (
            self._pixel_res - 1
        )  # quantized values inbetween 0 and 1

        # material conditioning vectors
        if config.cond:
            if config.raw:
                raise NotImplementedError("conditioning for raw is not supported")
            B, T, C, _, img_size = images.shape
            material_types = np.array(
                [t.observations["material"][0] for t in trajectories], dtype=int
            )
            num_cats = np.max(material_types) + 1

            # one-hot trick: stackoverflow.com/questions/38592324
            images_cat = np.zeros((B, T, C + num_cats, img_size, img_size))

            one_hot_vectors = np.eye(num_cats)[material_types]  # (B, num_cats)
            one_hot_vectors = one_hot_vectors[:, None, :, None, None]
            one_hot_vectors = np.tile(one_hot_vectors, (1, T, 1, img_size, img_size))

            images_cat[:, :, :C, :, :] = images
            images_cat[:, :, C:, :, :] = one_hot_vectors

            images = images_cat

        time = np.arange(0.0, images[0].shape[0]) * self.dt
        if config.raw:
            self._data = [
                (time, states[i], controls[i]) for i in range(total_trajectories)
            ]

            self._viz_data = self._data[-config.num_viz_trajectories :]
            self._data = self._data[: -config.num_viz_trajectories]
            self._data = [
                (
                    time[: config.traj_len],
                    image[: config.traj_len],  # img data
                    control[: config.traj_len],  # ctrl inputs
                )
                for time, image, control in self._data
            ]
        else:
            self._data = [
                (
                    time,
                    images[i],  # img data
                    controls[i],  # ctrl inputs
                    states[i],  # original xy data
                )
                for i in range(total_trajectories)
            ]

            self._viz_data = self._data[-config.num_viz_trajectories :]
            self._data = self._data[: -config.num_viz_trajectories]
            self._data = [
                (
                    time[: config.traj_len],
                    image[: config.traj_len],  # img data
                    control[: config.traj_len],  # ctrl inputs
                    state[: config.traj_len],  # original xy data
                )
                for time, image, control, state in self._data
            ]

    def get_viz_data(self, device: torch.device) -> VisData:
        """Get a VisData object of the viz dataset."""
        if self._config.raw:
            return _get_viz_data_basic(self, device)

        t_list = [t for t, x, u, p in self._viz_data]
        x_list = [x for t, x, u, p in self._viz_data]
        u_list = [u for t, x, u, p in self._viz_data]
        p_list = [p for t, x, u, p in self._viz_data]

        t = torch.tensor(t_list, dtype=torch.float, device=device)[0, :]
        y = torch.tensor(x_list, dtype=torch.float, device=device).transpose(0, 1)
        u = torch.tensor(u_list, dtype=torch.float, device=device).transpose(0, 1)

        np_p = np.array(p_list)
        np_v = np.gradient(np_p, axis=1) / self.dt
        pv = torch.tensor(
            np.concatenate((np_p, np_v), axis=-1), dtype=torch.float, device=device,
        ).transpose(0, 1)
        return VisDataIMG(
            t=t,
            y0=y[0],
            y=y,
            u=u,
            np_t=t.clone().cpu().numpy(),
            np_y=y.clone().cpu().numpy(),
            np_u=u.clone().cpu().numpy(),
            plot_settings=self.get_default_plot_settings(),
            pv=pv,
            np_pv=pv.clone().cpu().numpy(),
        )

    def get_default_plot_settings(self) -> PlotSettings:
        """Get plot settings for visualizations."""
        return PlotSettings(axis=Axis(xlim=(-3.0, 3.0), ylim=(-3.0, 3.0)))


@dataclass
class MITPushDatasetConfig(DatasetConfig):
    """MIT Push dataset configuration class."""

    half_image_size: bool
    pixel_res: int = 256
    raw: bool = False
    materials: Tuple[int, ...] = (0, 1, 2, 3)
    cond: bool = False
    kloss_dataset: bool = True

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Observation dimension."""
        if self.raw:
            return (2,)
        else:
            return (1, 16, 16) if self.half_image_size else (1, 32, 32)

    def create(self) -> "DynamicsDataset":
        """Create a `DynamicsDataset`."""
        dataset = MITPushDataset(self)
        return dataset


def _load_trajectories(
    *input_files,
    use_vision: bool = True,
    use_proprioception: bool = True,
    use_haptics: bool = True,
    vision_interval: int = 10,
    image_blackout_ratio: float = 0.0,
    sequential_image_rate: int = 1,
    start_timestep: int = 0,
    kloss_dataset: bool = True,
) -> List[TrajectoryNumpy]:
    """Loads a list of trajectories.

    credit:  https://github.com/brentyi/multimodalfilter/blob/4378d51fd847b4909ba60d8ac4ea183607afc7d4/crossmodal/tasks/_push.py

    Trajectories are from a set of input files, where each trajectory is a tuple containing...
        states: an (T, state_dim) array of state vectors
        observations: a key->(T, *) dict of observations
        controls: an (T, control_dim) array of control vectors
    Each input can either be a string or a (string, int) tuple, where int indicates the
    maximum number of trajectories to import.

    Parameters
    ----------
    *input_files :
        Trajectory inputs. Should be members of `crossmodal.push_data.dataset_urls.keys()`.
    use_vision : bool
        Set to False to zero out camera inputs.
    vision_interval : int
        Number of times each camera image is duplicated. For emulating a slow image rate.
    use_proprioception : bool
        Set to False to zero out kinematics data.
    use_haptics : bool
        Set to False to zero out F/T sensors.
    image_blackout_ratio : float
        Dropout probabiliity for camera inputs. 0.0 = no dropout, 1.0 = all images dropped out.
    sequential_image_rate : int
        If value is `N`, we only send 1 image frame ever `N` timesteps. All others are zeroed out.
    start_timestep : int
        If value is `N`, we skip the first `N` timesteps of each trajectory.

    Returns
    -------
    List[TrajectoryNumpy]
        List of trajectories.
    """
    if kloss_dataset:
        dataset_urls: Dict[str, str] = {
            "kloss_train0.hdf5": "https://drive.google.com/file/d/1nk4BO0rcVTKw22vYq6biewiwAFUPevM1/view?usp=sharing",
            "kloss_train1.hdf5": "https://drive.google.com/file/d/1nk4BO0rcVTKw22vYq6biewiwAFUPevM1/view?usp=sharing",
            "kloss_train2.hdf5": "https://drive.google.com/file/d/15W2zj52bSITxIRVRi7ajehAmz14RU33M/view?usp=sharing",
            "kloss_train3.hdf5": "https://drive.google.com/file/d/1WhRFu4SDlIYKnLYLyDdgOQYjP20JOTLE/view?usp=sharing",
            "kloss_train4.hdf5": "https://drive.google.com/file/d/1-ur_hzyBvd1_QCLTamaO8eWJ7rXii7y4/view?usp=sharing",
            "kloss_train5.hdf5": "https://drive.google.com/file/d/1ni8vEy4c1cmCKP2ZlWfXqLo7a4sdRFwe/view?usp=sharing",
            "kloss_val.hdf5": "https://drive.google.com/file/d/1-CRocf7I4mTLBp7Tjo7-D-QvkwcGZkNo/view?usp=sharing",
        }
    else:
        dataset_urls: Dict[str, str] = {
            "gentle_push_10.hdf5": "https://drive.google.com/file/d/1qmBCfsAGu8eew-CQFmV1svodl9VJa6fX/view?usp=sharing",
            "gentle_push_100.hdf5": "https://drive.google.com/file/d/1PmqQy5myNXSei56upMy3mXKu5Lk7Fr_g/view?usp=sharing",
            "gentle_push_300.hdf5": "https://drive.google.com/file/d/18dr1z0N__yFiP_DAKxy-Hs9Vy_AsaW6Q/view?usp=sharing",
            "gentle_push_1000.hdf5": "https://drive.google.com/file/d/1JTgmq1KPRK9HYi8BgvljKg5MPqT_N4cR/view?usp=sharing",
        }

    trajectories = []

    assert 1 > image_blackout_ratio >= 0
    assert image_blackout_ratio == 0 or sequential_image_rate == 1

    for name in input_files:
        max_trajectory_count = sys.maxsize
        if type(name) == tuple:
            name, max_trajectory_count = name
        assert type(max_trajectory_count) == int

        # Load trajectories file into memory, all at once
        with fannypack.data.TrajectoriesFile(
            fannypack.data.cached_drive_file(name, dataset_urls[name])
        ) as f:
            raw_trajectories = list(f)

        # Iterate over each trajectory
        for raw_trajectory_index, raw_trajectory in enumerate(raw_trajectories):
            if raw_trajectory_index >= max_trajectory_count:
                break

            if kloss_dataset:
                timesteps = len(raw_trajectory["pos"])
            else:
                timesteps = len(raw_trajectory["object-state"])

            # State is just (x, y)
            state_dim = 2
            states = np.full((timesteps, state_dim), np.nan)

            if kloss_dataset:
                states[:, 0] = raw_trajectory["pos"][:, 0]
                states[:, 1] = raw_trajectory["pos"][:, 2]
            else:
                states[:, :2] = raw_trajectory["Cylinder0_pos"][:, :2]  # x, y

            # Pull out observations
            # This currently consists of:
            # > gripper_pos: end effector position
            # > gripper_sensors: F/T, contact sensors
            # > image: camera image

            observations = {}

            if kloss_dataset:
                observations["gripper_pos"] = raw_trajectory["tip"]
            else:
                observations["gripper_pos"] = raw_trajectory["eef_pos"]
            assert observations["gripper_pos"].shape == (timesteps, 3)

            if kloss_dataset:
                observations["gripper_sensors"] = np.zeros((timesteps, 7))
                observations["gripper_sensors"][:, :3] = raw_trajectory["force"]
                observations["gripper_sensors"][:, 6] = raw_trajectory["contact"]
            else:
                observations["gripper_sensors"] = np.concatenate(
                    (
                        raw_trajectory["force"],
                        raw_trajectory["contact"][:, np.newaxis],
                    ),
                    axis=1,
                )
            assert observations["gripper_sensors"].shape[1] == 7

            # Zero out proprioception or haptics if unused
            if not use_proprioception:
                observations["gripper_pos"][:] = 0
            if not use_haptics:
                observations["gripper_sensors"][:] = 0

            # Get material
            if kloss_dataset:
                if raw_trajectory["material"].rstrip("\x00") == "pu":
                    observations["material"] = np.ones(timesteps) * 0
                elif raw_trajectory["material"].rstrip("\x00") == "abs":
                    observations["material"] = np.ones(timesteps) * 1
                elif raw_trajectory["material"].rstrip("\x00") == "plywood":
                    observations["material"] = np.ones(timesteps) * 2
                elif raw_trajectory["material"].rstrip("\x00") == "delrin":
                    observations["material"] = np.ones(timesteps) * 3
                else:
                    raise NotImplementedError(raw_trajectory["material"])

            # get contact point and normal vector
            if kloss_dataset:
                observations["contact_point"] = raw_trajectory["contact_point"]
                observations["normal"] = raw_trajectory["normal"]

            # Get image
            if kloss_dataset:
                observations["image"] = np.mean(raw_trajectory["image"], axis=-1)
            else:
                observations["image"] = raw_trajectory["image"].copy()
            assert observations["image"].shape == (timesteps, 32, 32)

            # Mask image observations based on dataset args
            image_mask: np.ndarray
            if not use_vision:
                # Use a zero mask
                image_mask = np.zeros((timesteps, 1, 1), dtype=np.float32)
            elif image_blackout_ratio == 0.0:
                # Apply sequential rate
                image_mask = np.zeros((timesteps, 1, 1), dtype=np.float32)
                image_mask[::sequential_image_rate, 0, 0] = 1.0
            else:
                # Apply blackout rate
                image_mask = (
                    (np.random.uniform(size=(timesteps,)) > image_blackout_ratio)
                    .astype(np.float32)
                    .reshape((timesteps, 1, 1))
                )
            observations["image"] *= image_mask

            # Pull out controls
            # This currently consists of:
            # > previous end effector position
            # > end effector position delta
            # > binary contact reading
            if kloss_dataset:
                eef_positions = raw_trajectory["tip"]
            else:
                eef_positions = raw_trajectory["eef_pos"]
            eef_positions_shifted = np.roll(eef_positions, shift=1, axis=0)
            eef_positions_shifted[0] = eef_positions[0]
            controls = np.concatenate(
                [
                    eef_positions_shifted,
                    eef_positions - eef_positions_shifted,
                    raw_trajectory["contact"][
                        :, np.newaxis
                    ],  # "contact" key same for both kloss and normal dataset
                ],
                axis=1,
            )
            assert controls.shape == (timesteps, 7)

            # Normalize data
            if kloss_dataset:
                observations["gripper_pos"] -= np.array(
                    [[-0.00360131, 0.0, 0.00022349]], dtype=np.float32
                )
                observations["gripper_pos"] /= np.array(
                    [[0.07005621, 1.0, 0.06883541]], dtype=np.float32
                )
                observations["gripper_sensors"] -= np.array(
                    [
                        [
                            3.04424347e-02,
                            1.61328610e-02,
                            -2.47517393e-04,
                            0.00000000e00,
                            0.00000000e00,
                            0.00000000e00,
                            # 6.25842857e-01,
                            0.00000000e00,  # adjusting the "normalization" for contact
                        ]
                    ]
                )
                observations["gripper_sensors"] /= np.array(
                    # [[2.09539968, 2.0681382, 0.00373115, 1.0, 1.0, 1.0, 0.48390451]]
                    [[2.09539968, 2.0681382, 0.00373115, 1.0, 1.0, 1.0, 1.0]]
                )
                states -= np.array([[-0.00279736, -0.00027878]])
                states /= np.array([[0.06409658, 0.06649422]])
                controls -= np.array(
                    [
                        [
                            -3.55868486e-03,
                            0.00000000e00,
                            2.34369027e-04,
                            -4.26185595e-05,
                            0.00000000e00,
                            -1.08724583e-05,
                            6.25842857e-01,
                        ]
                    ]
                )
                controls /= np.array(
                    [
                        [
                            0.0693582,
                            1.0,
                            0.06810329,
                            0.01176415,
                            1.0,
                            0.0115694,
                            0.48390451,
                        ]
                    ]
                )

            else:
                observations["gripper_pos"] -= np.array(
                    [[0.46806443, -0.0017836, 0.88028437]], dtype=np.float32
                )
                observations["gripper_pos"] /= np.array(
                    [[0.02410769, 0.02341035, 0.04018243]], dtype=np.float32
                )
                observations["gripper_sensors"] -= np.array(
                    [
                        [
                            4.9182904e-01,
                            4.5039989e-02,
                            -3.2791464e00,
                            -3.3874984e-03,
                            1.1552566e-02,
                            -8.4817986e-04,
                            # 2.1303751e-01,
                            0.0,
                        ]
                    ],
                    dtype=np.float32,
                )
                observations["gripper_sensors"] /= np.array(
                    [
                        [
                            1.6152629,
                            1.666905,
                            1.9186896,
                            0.14219016,
                            0.14232528,
                            0.01675198,
                            # 0.40950698,
                            1.0,
                        ]
                    ],
                    dtype=np.float32,
                )
                states -= np.array([[0.4970164, -0.00916641]])
                states /= np.array([[0.0572766, 0.06118315]])
                controls -= np.array(
                    [
                        [
                            4.6594709e-01,
                            -2.5247163e-03,
                            8.8094306e-01,
                            1.2939950e-04,
                            -5.4364675e-05,
                            -6.1112235e-04,
                            # 2.2041667e-01,
                            0.0,
                        ]
                    ],
                    dtype=np.float32,
                )
                controls /= np.array(
                    [
                        [
                            0.02239027,
                            0.02356066,
                            0.0405312,
                            0.00054858,
                            0.0005754,
                            0.00046352,
                            # 0.41451886,
                            1.0,
                        ]
                    ],
                    dtype=np.float32,
                )

            # downsample if we use the mujoco dataset
            if kloss_dataset:
                trajectories.append(
                    TrajectoryNumpy(
                        states[start_timestep:],
                        fannypack.utils.SliceWrapper(observations)[start_timestep:],
                        controls[start_timestep:],
                    )
                )
            else:
                start_timestep = 180  # DEBUG: just hardcode this here
                trajectories.append(
                    TrajectoryNumpy(
                        states[start_timestep::2],
                        fannypack.utils.SliceWrapper(observations)[start_timestep::2],
                        controls[start_timestep::2],
                    )
                )

            # Reduce memory usage
            raw_trajectories[raw_trajectory_index] = None
            del raw_trajectory

    return trajectories
