from dataclasses import dataclass
from typing import Callable, Optional

from dynamics_learning.networks.estimator import EstimatorConfig

LearningRateScheduler = Callable[[int, float], float]


@dataclass(frozen=True)
class ExpConfig:
    """All parameters for training.

    This must be serializable.
    """

    model: EstimatorConfig
    ramp_iters: int
    batch_size: int
    epochs: int
    base_learning_rate: float
    gradient_clip_max_norm: Optional[float] = None
    const_var: bool = False
    log_iterations_simple: int = 50
    log_iterations_images: int = 100
    git_commit_hash: Optional[str] = None
    learning_rate_function: Optional[LearningRateScheduler] = None
    name: Optional[str] = None
