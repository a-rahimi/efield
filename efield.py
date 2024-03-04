from typing import List, NamedTuple

import matplotlib.pylab as plt
import torch
from torch import nn


def potential_operator(
    locations3d: torch.Tensor,
    anchor_locations3d: torch.Tensor,
    anchor_widths: torch.Tensor,
) -> torch.Tensor:
    if locations3d.shape[1] != 3:
        raise ValueError("Locations3d must be an Nx3 tensor")

    D = torch.cdist(locations3d.unsqueeze(0), anchor_locations3d.unsqueeze(0)).squeeze()
    return torch.exp(-anchor_widths * D**2)


def field_operator(
    locations3d: torch.Tensor,
    anchor_locations3d: torch.Tensor,
    anchor_widths: torch.Tensor,
) -> torch.Tensor:
    """A tensor that converts coeffients to the derivative of the field wrt the
    coordinate axes at the given locations.

    If f(x) is the potential function (as implemented in forward()), this
    function returns df/(dx danchor_weights).

    Returns a 3x len(locations3d) x len(anchor_locations) tensor.
    """
    k = potential_operator(locations3d, anchor_locations3d, anchor_widths)

    # The derivative of k wrt the coordinat axes.
    return (
        -2
        * anchor_widths
        * k
        * torch.stack(
            (
                locations3d[:, 0][:, None] - anchor_locations3d[:, 0][None, :],
                locations3d[:, 1][:, None] - anchor_locations3d[:, 1][None, :],
                locations3d[:, 2][:, None] - anchor_locations3d[:, 2][None, :],
            )
        )
    )


class Potential(nn.Module):
    """Map a point in three-dimensional space to a scalar potential."""

    def __init__(
        self,
        anchor_locations3d: torch.Tensor,
        anchor_weights: torch.Tensor,
        anchor_widths: torch.Tensor,
    ):
        super().__init__()
        self.anchor_locations3d = nn.Parameter(anchor_locations3d)
        self.anchor_weights = nn.Parameter(anchor_weights)
        self.anchor_widths = nn.Parameter(anchor_widths)

    def forward(self, locations3d: torch.Tensor) -> torch.Tensor:
        return (
            potential_operator(locations3d, self.anchor_locations3d, self.anchor_widths)
            @ self.anchor_weights
        )

    def field(self, locations3d: torch.Tensor) -> torch.Tensor:
        return (
            field_operator(locations3d, self.anchor_locations3d, self.anchor_widths)
            @ self.anchor_weights
        )

    def enclosed_charge(sphere_center3d: torch.Tensor, sphere_radius: float) -> float:
        raise NotImplementedError()


def minimum_norm_in_subspace(
    A: torch.Tensor, B: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    "Minimize ||Ax|| subject to Bx=b over x."

    if B.shape[0] > B.shape[1]:
        raise ValueError("The constrained must be under-determined")

    # The constraint ipmlies that x must have the form
    #   x = x0 - Z u
    # where x0 = B\b, and Z is the null space of B.
    U, s, Vt = torch.linalg.svd(B, full_matrices=True)
    x0 = Vt[: U.shape[0]].T @ (U.T / s[:, None]) @ b
    Z = Vt[U.shape[0] :, :].T

    # The objective, in terms of u, is
    #
    #   ||Ax||^2 = ||A (x0 - Z u)||^2
    #
    # Differenting wrt to u and setting to zero gives
    #
    #   0 = Z'A'A (x0 - Z u)
    #
    # So
    #
    #  Z'A'AZ u = Z'A'A x0, or
    #
    # equivalently,
    #
    #    u = AZ \ A x0
    u = torch.linalg.lstsq(A @ Z, A @ x0)[0]

    return x0 - Z @ u


def test_minimum_L2_norm_in_subspace():
    B = torch.randn(3, 5)
    b = torch.rand(3)

    x = minimum_norm_in_subspace(torch.eye(5), B, b)

    x_true = torch.linalg.lstsq(B, b)[0]

    assert torch.allclose(B @ x, b)
    assert torch.allclose(x, x_true)


test_minimum_L2_norm_in_subspace()


def test_pair_of_points():
    # Solve the problem
    #   min ||x1 - x2|| s.t x1 + x2 = 2
    # A solution to this is obviously x1=x2=1, since it achieves 0 and satisfies
    # the constraints. It is also the only minimizer because the constaint imlies x1=2-x2, so the
    # objective is ||2-2 x2||, which has exactly one minimizer.

    x = minimum_norm_in_subspace(
        torch.tensor([[1.0, -1.0]]), torch.tensor([[1.0, 1.0]]), torch.tensor([2.0])
    )
    assert torch.allclose(x, torch.tensor([1.0, 1.0]))


test_pair_of_points()


class Box(NamedTuple):
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def grid(self, nx: int, ny: int, nz: int) -> torch.Tensor:
        return (
            torch.stack(
                torch.meshgrid(
                    torch.linspace(self.xmin, self.xmax, nx),
                    torch.linspace(self.ymin, self.ymax, ny),
                    torch.linspace(self.zmin, self.zmax, nz),
                    indexing="xy",
                )
            )
            .reshape(3, -1)
            .T
        )


def potential_function(
    test_locations3d: torch.Tensor,
    conductor_locations3d: torch.Tensor,
    conductor_potentials: torch.Tensor,
    verbose=True,
) -> Potential:
    # Solve the constrained optimization problem
    #
    #   min_f \int_x ||d/dx f(x)||^2
    #   s.t.  f(x) = V(x) when for all x on the conductors.

    class Result(NamedTuple):
        field_energy: float
        constraint_violation: float
        potential: Potential

    results: List[Result] = []

    baseline_anchor_width = (torch.cdist(test_locations3d, test_locations3d)).mean()
    for anchor_width in (torch.linspace(0.1, 1, 20) * baseline_anchor_width) ** 2:
        # A 3*N x N matrix that maps coefficients to the electric field. The
        # field is represented as a 3*N long vector.
        DK = field_operator(test_locations3d, test_locations3d, anchor_width).reshape(
            -1, len(test_locations3d)
        )
        # An M x N matrix that maps coefficients to voltage on the M conductors.
        K = potential_operator(conductor_locations3d, test_locations3d, anchor_width)

        coeffs = minimum_norm_in_subspace(DK, K, conductor_potentials)

        results.append(
            Result(
                torch.norm(DK @ coeffs).item(),
                torch.norm(K @ coeffs - conductor_potentials).item()
                / len(conductor_potentials),
                Potential(test_locations3d, coeffs, anchor_width),
            )
        )

    if verbose:
        anchor_widths = [r.potential.anchor_widths.item() for r in results]

        _, ax = plt.subplots(1, 1)

        best_result = min(results)
        ax.axhline(best_result.field_energy, ls="--", lw=1)
        ax.axvline(best_result.potential.anchor_widths.item(), ls="--", lw=1)

        ax.plot(anchor_widths, [r.field_energy for r in results], color="c")
        ax.set_xlabel("anchor_width")
        ax.set_ylabel("Field energy", color="c")
        ax.tick_params(axis="y", labelcolor="c")

        ax2 = ax.twinx()
        ax2.plot(
            anchor_widths,
            [r.constraint_violation for r in results],
            color="r",
        )
        ax2.set_ylabel("Constraing violation norm", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

    return best_result.potential
