import numpy as np


def policy1(z: np.ndarray, t: float) -> np.ndarray:
    """Pseudo-random open-loop policy."""
    _dt = 0.025
    return np.array([2 * np.sin(_prng(z, t, _dt))])


# linear congruential prng (only depends on time)
# > stackoverflow.com/questions/3062746
def _prng(z: np.ndarray, t: float, dt: float) -> int:
    """PRNG method for control."""
    _t = dt * ((t / dt) // 1)
    m = 2 ** 31
    a = 1103515245
    c = 12345
    albert_num = int((1221341234932874 * (1 + _t)) // 1)
    seed = (a * albert_num + c) % m
    return (a * seed + c) % m
