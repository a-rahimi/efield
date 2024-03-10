import torch

import efield


def test_pairwise_squared_distances():
    A = torch.rand(3, 2)
    B = torch.rand(4, 2)

    D_expected = torch.cdist(A.unsqueeze(0), B.unsqueeze(0)).squeeze() ** 2
    D_actual = efield.pairwise_squared_distances(A, B)
    assert torch.allclose(D_expected, D_actual, atol=1e-4), (D_expected, D_actual)


test_pairwise_squared_distances()


def test_minimum_norm_in_subspace_minimum_L2_norm_in_subspace():
    A = torch.randn(3, 5)
    b = torch.rand(3)

    x = efield.minimum_norm_in_subspace(torch.eye(5), A, b)

    x_true = torch.linalg.lstsq(A, b)[0]

    assert torch.allclose(A @ x, b, rtol=1e-3)
    assert torch.allclose(x, x_true)


test_minimum_norm_in_subspace_minimum_L2_norm_in_subspace()


def test_minimum_norm_in_subspace_pair_of_points():
    # Solve the problem
    #
    #   min ||x1||^2 + ||x1 - x2||^2 s.t x1 + x2 = 2
    #
    # The minimizer is x1=x2=1. The constaint implies x1 = 2 - x2, so the objective
    # is ||2 - x2||^2 + ||2 - 2 x2||^2. The minimizer of this satisfies
    #
    #    0 = -(2 - x2) - 2 (2 - 2 x2)
    #
    # or
    #
    #    6 = 5 x2.

    M = torch.outer(torch.tensor([1.0, -1.0]), torch.tensor([1.0, -1.0]))
    M[0, 0] += 1
    x = efield.minimum_norm_in_subspace(
        M, torch.tensor([[1.0, 1.0]]), torch.tensor([2.0])
    )

    assert torch.allclose(x, torch.tensor((2 - 6 / 5, 6 / 5))), x


test_minimum_norm_in_subspace_pair_of_points()


def test_quadratic_form():
    x = torch.randn(3)
    x1 = torch.randn(3)
    x2 = torch.randn(3)
    assert torch.allclose(
        torch.linalg.norm(x - x1) ** 2 + torch.linalg.norm(x - x2) ** 2,
        2 * torch.linalg.norm(x - (x1 + x2) / 2) ** 2
        + 0.5 * torch.linalg.norm(x1 - x2) ** 2,
    )


test_quadratic_form()


def test_quadratic_form_expectation():
    ndim = 3
    x1 = torch.rand(ndim)
    x2 = torch.rand(ndim)

    mu = (x1 + x2) / 2
    sigma = 0.3
    x = mu + sigma * torch.randn(1_000_000, ndim)

    empirical = torch.mean(torch.sum((x - x1) * (x - x2), axis=1))
    analytical = ndim * sigma**2 - torch.linalg.norm(x1 - x2) ** 2 / 4

    assert torch.allclose(empirical, analytical, atol=1e-3)


test_quadratic_form_expectation()


def test_field_energy():
    sigma = 1.3
    anchor_locations3d = torch.randn(4, 3)

    # Compute the energy in the field by evaluating the field at a set of points
    # randomly drawn from the interior of a 3D box with edge length box_width, then
    # everaging the norm of the gradient of this field over these random draws.
    box_width = 10
    xyz = box_width * (torch.rand(10_000_000, 3) - 0.5)
    Vxyz = efield.field_operator(xyz, anchor_locations3d, sigma).sum(axis=2)
    empirical = (Vxyz**2).sum() * box_width**3 / len(xyz)

    # Compare against the analytic solutino derived in simulator.md
    D2 = efield.pairwise_squared_distances(anchor_locations3d, anchor_locations3d)
    analytical = (
        torch.sum(torch.exp(-D2 / sigma**2 / 4) * (3 / 2 * sigma**2 - D2 / 4))
        / sigma
        * torch.pi ** (3 / 2)
    )

    assert torch.allclose(analytical, empirical, rtol=1e-2), (analytical, empirical)


test_field_energy()
