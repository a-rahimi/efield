import numpy as np
import torch

import efield
import geometry


def test_minimum_norm_in_subspace_minimum_L2_norm_in_subspace():
    # Test efield.minimum_norm_in_subspace by checking that when we
    # ask it solve the problem
    #   min ||x|| s.t. Ax=b
    # it returns just A\b.
    A = torch.randn(3, 5)
    b = torch.rand(3)

    x = efield.minimum_norm_in_subspace(torch.eye(5), A, b)

    x_true = torch.linalg.lstsq(A, b)[0]

    assert torch.allclose(A @ x, b, rtol=1e-3)
    assert torch.allclose(x, x_true)


def test_minimum_norm_in_subspace_pair_of_points():
    # Test efield.minimum_norm_in_subspace by checking its solution to
    # the problem
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
    #    6 = 5 * x2.

    M = torch.outer(torch.tensor([1.0, -1.0]), torch.tensor([1.0, -1.0]))
    M[0, 0] += 1
    x = efield.minimum_norm_in_subspace(
        M, torch.tensor([[1.0, 1.0]]), torch.tensor([2.0])
    )

    assert torch.allclose(x, torch.tensor((2 - 6 / 5, 6 / 5))), x


def test_quadratic_form():
    # Check that
    #
    #   ||x-x1||^2 + ||x-x1||^2 =  2 * ||x-(x1+x2)/2||^2 + 1/2 * ||x1-x2||^2.
    #
    # This result is used in simulator.md.
    x = torch.randn(3)
    x1 = torch.randn(3)
    x2 = torch.randn(3)
    assert torch.allclose(
        torch.linalg.norm(x - x1) ** 2 + torch.linalg.norm(x - x2) ** 2,
        2 * torch.linalg.norm(x - (x1 + x2) / 2) ** 2
        + 0.5 * torch.linalg.norm(x1 - x2) ** 2,
    )


def test_quadratic_form_expectation():
    # Check that the expected value of (x-x1)'(x-x2) when x ~ N(mu,sigma^2)
    # is 3 sigma^2 - ||x1-x2||^2/4.
    ndim = 3
    x1 = torch.rand(ndim)
    x2 = torch.rand(ndim)

    mu = (x1 + x2) / 2
    sigma = 0.3
    x = mu + sigma * torch.randn(1_000_000, ndim)

    empirical = torch.mean(torch.sum((x - x1) * (x - x2), axis=1))
    analytical = ndim * sigma**2 - torch.linalg.norm(x1 - x2) ** 2 / 4

    assert torch.allclose(empirical, analytical, atol=1e-3)


def test_field_energy():
    # Confirm the quadratic form that evaluates to the energy of a field.

    sigma = 1.3
    anchor_locations3d = torch.randn(4, 3)

    # Compute the energy in the field by evaluating the field at a set of points
    # randomly drawn from the interior of a 3D box with edge length box_width, then
    # everaging the norm of the gradient of this field over these random draws.
    box_width = 10
    xyz = box_width * (torch.rand(10_000_000, 3) - 0.5)
    Vxyz = efield.RadialBasisFunctionOperators.field_operator(
        xyz, anchor_locations3d, sigma
    ).sum(axis=2)
    empirical = (Vxyz**2).sum() * box_width**3 / len(xyz)

    # Compare against the analytic solutino derived in simulator.md
    D2 = geometry.pairwise_squared_distances(anchor_locations3d, anchor_locations3d)
    analytical = (
        torch.sum(torch.exp(-D2 / sigma**2 / 4) * (3 / 2 * sigma**2 - D2 / 4))
        / sigma
        * torch.pi ** (3 / 2)
    )

    assert torch.allclose(analytical, empirical, rtol=1e-2), (analytical, empirical)


def test_integral_of_gaussian():
    # Compare the output efield.integral_of_gaussian against a numerical
    # integration.
    mean = 0.5
    std = 2
    xmin = 1.1
    xmax = 1.8
    actual = efield.integral_of_gaussian(
        torch.tensor(mean), torch.tensor(std), torch.tensor(xmin), torch.tensor(xmax)
    )

    x = np.linspace(xmin, xmax, 10_000)
    expected = sum(np.exp(-0.5 * (x - mean) ** 2 / std**2)) * (x[1] - x[0])

    assert np.allclose(actual, expected, rtol=1e-4), (actual, expected)


def random_radial_basis_function_potential() -> efield.RadialBasisFunctionPotential:
    num_anchors = 10
    return efield.RadialBasisFunctionPotential(
        anchor_locations3d=torch.randn(num_anchors, 3),
        anchor_coeffs=torch.rand(num_anchors),
        anchor_parameters=torch.tensor(3.0),
    )


def test_flux_through_face():
    # Compare the closed form flux through a rectangular face against a
    # numerically integrated answer.

    potential = random_radial_basis_function_potential()

    face = geometry.Face(
        geometry.AxisAlignedInterval(0, -1, +1),
        geometry.AxisAlignedInterval(1, -0.5, +0.5),
        offset_axis=2,
        offset=0,
        orientation=1,
    )
    # Compute the flux through the face using the closed form solution.
    actual_flux = potential.flux_through_face(face)

    # Compute the flux through the face with numerical quadrature.
    X, Y = torch.meshgrid(
        torch.linspace(face.first_axis.min, face.first_axis.max, 200),
        torch.linspace(face.second_axis.min, face.second_axis.max, 200),
        indexing="xy",
    )
    Z = torch.zeros_like(X) + face.offset
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    face_xyz = torch.stack((X.flatten(), Y.flatten(), Z.flatten())).T
    field = potential.field(face_xyz)

    expected_flux = (field[face.offset_axis, :] * face.orientation).sum() * dx * dy

    assert torch.allclose(expected_flux, actual_flux, rtol=1e-2), (
        expected_flux,
        actual_flux,
    )
