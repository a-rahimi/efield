import numpy as np
import torch

import efield
import geometry


def test_affine_projection_is_feasiable():
    # Confirm that after projecting a point, it actually satisfies the
    # constraint.
    A = torch.randn(3, 5)
    b = torch.rand(3)
    offset, proj = efield.affine_projection(A, b)

    x = torch.rand(5)
    x_proj = proj @ x + offset
    assert torch.allclose(A @ x_proj, b)


def test_affine_projection_first_coordinate():
    # Find the projection operator when A just picks the first entry
    # of x. So we're solving
    #   min_u ||u-x|| s.t. u_1 = 1
    offset, proj = efield.affine_projection(
        torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([1.0])
    )

    # The solution just replace x_1 with 1.
    assert torch.allclose(
        proj, torch.tensor([[0.0, 0.0, 0.0], [0, 1, 0], [0, 0, 1]])
    ), proj
    assert torch.allclose(offset, torch.tensor([1.0, 0.0, 0.0])), offset
