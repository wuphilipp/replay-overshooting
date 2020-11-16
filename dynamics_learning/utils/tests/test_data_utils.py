import numpy as np

from dynamics_learning.utils.data_utils import GaussianSampler, UniformSampler
from dynamics_learning.utils.plot_utils import PlotHandler


def test_UniformSampler():
    sampler_y0 = UniformSampler(np.array([-10, -10]), np.array([10, 10]))
    batch = sampler_y0.sample_batch(4)
    assert batch.shape == (4, 2)


def test_GaussianSampler():
    sampler_y0 = GaussianSampler(np.array([-1, -2]), np.array([[10, 1], [1, 10]]))
    batch = sampler_y0.sample_batch(4)
    assert batch.shape == (4, 2)


def test_plot_handler_runs():
    ph = PlotHandler
    with ph.plot_context() as (fig, ax):
        data1 = np.random.normal(size=(20, 100, 2))
        ph.plot_xy(data1)

    with ph.plot_context() as (fig, ax):
        data1 = np.random.normal(size=(20, 100, 2))
        data2 = np.random.normal(size=(20, 100, 2))
        ph.plot_xy_compare([data1, data2])
        ph.plot_as_image(fig)
