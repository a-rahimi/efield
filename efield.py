from typing import List, NamedTuple

import matplotlib.pylab as plt
import torch
from torch import nn


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


def potential_operator(
    locations3d: torch.Tensor,
    anchor_locations3d: torch.Tensor,
    anchor_scale: torch.Tensor,
) -> torch.Tensor:
    if locations3d.shape[1] != 3:
        raise ValueError("Locations3d must be an Nx3 tensor")
    if anchor_locations3d.shape[1] != 3:
        raise ValueError("anchor_locations3d must be an Mx3 tensor")

    D = torch.cdist(locations3d.unsqueeze(0), anchor_locations3d.unsqueeze(0)).squeeze()
    assert D.shape == (len(locations3d), len(anchor_locations3d))
    return torch.exp(-anchor_scale * D**2)


def field_operator(
    locations3d: torch.Tensor,
    anchor_locations3d: torch.Tensor,
    anchor_scale: torch.Tensor,
) -> torch.Tensor:
    """A tensor that converts coeffients to the derivative of the field wrt the
    coordinate axes at the given locations.

    If f(x) is the potential function (as implemented in forward()), this
    function returns df/(dx danchor_weights).

    Returns a 3x len(locations3d) x len(anchor_locations) tensor.
    """
    k = potential_operator(locations3d, anchor_locations3d, anchor_scale)

    # The derivative of k wrt the coordinat axes.
    return (
        -2
        * anchor_scale
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
            potential_operator(
                locations3d, self.anchor_locations3d, 1 / self.anchor_widths**2
            )
            @ self.anchor_weights
        )

    def field(self, locations3d: torch.Tensor) -> torch.Tensor:
        return (
            field_operator(
                locations3d, self.anchor_locations3d, 1 / self.anchor_widths**2
            )
            @ self.anchor_weights
        )

    def enclosed_charge(enclosure: Box) -> float:
        raise NotImplementedError()


class IllConditionedMatrixError(ValueError):
    def __init__(self, msg: str):
        super().__init__(msg)


def minimum_norm_in_subspace(
    M: torch.Tensor, K: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Minimize x'Mx subject to Ax=b over x.

    See simulatord.md for a derivation.
    """
    if K.shape[0] > K.shape[1]:
        raise ValueError("The constraint must be under-determined")

    V = torch.linalg.solve(M, K.T)
    return V @ torch.linalg.solve(K @ V, b)


def test_minimum_L2_norm_in_subspace():
    A = torch.randn(3, 5)
    b = torch.rand(3)

    x = minimum_norm_in_subspace(torch.eye(5), A, b)

    x_true = torch.linalg.lstsq(A, b)[0]

    assert torch.allclose(A @ x, b)
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


def potential_function(
    anchor_locations3d: torch.Tensor,
    conductor_locations3d: torch.Tensor,
    conductor_potentials: torch.Tensor,
    verbose=True,
) -> Potential:
    if not (anchor_locations3d.ndim == 2 and anchor_locations3d.shape[1] == 3):
        raise ValueError("test_locations3 must be Mx3 tensor.")
    if not (conductor_locations3d.ndim == 2 or conductor_locations3d.shape[1] == 3):
        raise ValueError("conductor_locations3d must be Nx3 tensor.")
    if conductor_potentials.ndim != 1:
        raise ValueError("condctuctor_potential must be an N-dimensional vector.")

    # Solve the constrained optimization problem
    #
    #   min_f \int_x ||d/dx f(x)||^2
    #   s.t.  f(x) = V(x) when for all x on the conductors.
    #
    # When f(x) = sum_i coeffs[i] exp(-||x-anchor[i]||^2/width^2),
    # this problem reduces to minimizing a quadratic form subject to linear
    # constraints. See simulation.md for an explanation.

    class Result(NamedTuple):
        field_energy: float
        constraint_violation: float
        potential: Potential

    results: List[Result] = []

    norms_sq = torch.linalg.norm(anchor_locations3d) ** 2

    baseline_anchor_width = torch.cdist(anchor_locations3d, anchor_locations3d).mean()
    for anchor_width in torch.linspace(0.5, 1.5, 10) * baseline_anchor_width:
        # A matrix M so that coeffs' M coeffs gives the energy of the field.
        M = (
            1
            / anchor_width
            * potential_operator(
                anchor_locations3d, anchor_locations3d, (0.5 / anchor_width) ** 2
            )
            * (anchor_width**2 - norms_sq[:, None] - norms_sq[None, :])
        )

        # A matrix that maps coefficients to voltage on the M conductors.
        K = potential_operator(
            conductor_locations3d, anchor_locations3d, 1 / anchor_width**2
        )

        try:
            coeffs = minimum_norm_in_subspace(M, K, conductor_potentials)
        except IllConditionedMatrixError as e:
            print("Skipping anchor_width %g. %s" % (anchor_width, e))
            continue

        results.append(
            Result(
                field_energy=(coeffs.T @ K @ coeffs).item(),
                constraint_violation=torch.norm(
                    K @ coeffs - conductor_potentials
                ).item()
                / len(conductor_potentials),
                potential=Potential(anchor_locations3d, coeffs, anchor_width),
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
