from typing import Optional

import torch

from dynamics_learning.networks.models import ODENet
from dynamics_learning.utils.tests.test_net_utils import assert_allclose_tensors


class LinearSystem(ODENet):
    """A linear system used for testing."""

    def __init__(
        self, input_dim: int, output_dim: int,
    ):
        """Create a linear system.

        Parameters
        ----------
        input_dim : int
            Number of input dimensions.
        output_dim : int
            Number of output dimensions.
        """
        super(LinearSystem, self).__init__()
        self.net = torch.nn.Linear(input_dim, output_dim)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        u: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """See parent class."""
        return self.net(x)


def assert_tensor_symmetric_psd(matrix: torch.Tensor) -> None:
    """Check if a matrix is symmetric PSD.

    Parameters
    ----------
    matrix : torch.Tensor
        A tensor to be checked.
    """
    assert matrix.shape[-1] == matrix.shape[-2]  # square

    symmetric_matrix = (matrix + matrix.transpose(-1, -2)) / 2
    assert_allclose_tensors(symmetric_matrix, matrix)  # symmetric

    eig_values, _ = torch.symeig(symmetric_matrix)
    assert (eig_values > 0).all()  # PSD
