from typing import List, NamedTuple

import matplotlib.pylab as plt
import torch
from torch import nn
import tqdm


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


def pairwise_squared_distances(X, Y):
    # D[i,j] = ||x[i]-y[j]||^2
    #        = ||x[i]||^2 + ||y[j]|| - 2x[i]'y[j]
    # So the matrix D is
    #    ||x||^2 + ||y||^2 - 2 x'y
    return (X**2).sum(axis=1)[:, None] + (Y**2).sum(axis=1)[None, :] - 2 * X @ Y.T


def potential_operator(
    locations3d: torch.Tensor,
    anchor_locations3d: torch.Tensor,
    anchor_width: torch.Tensor,
) -> torch.Tensor:
    if locations3d.shape[1] != 3:
        raise ValueError("Locations3d must be an Nx3 tensor")
    if anchor_locations3d.shape[1] != 3:
        raise ValueError("anchor_locations3d must be an Mx3 tensor")

    return torch.exp(
        -0.5
        * pairwise_squared_distances(locations3d, anchor_locations3d)
        / anchor_width**2
    )


def field_operator(
    locations3d: torch.Tensor,
    anchor_locations3d: torch.Tensor,
    anchor_width: torch.Tensor,
) -> torch.Tensor:
    """A tensor that converts coeffients to the derivative of the field wrt the
    coordinate axes at the given locations.

    If f(x) is the potential function (as implemented in forward()), this
    function returns df/(dx danchor_weights).

    Returns a 3x len(locations3d) x len(anchor_locations) tensor.
    """
    k = potential_operator(locations3d, anchor_locations3d, anchor_width)

    # The derivative of k wrt the coordinat axes.
    return (
        -k
        / anchor_width**2
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

    def enclosed_charge(enclosure: Box) -> float:
        raise NotImplementedError()


def solve(A, b):
    "Solve for x in A x = b."
    return torch.linalg.solve(A, b)
    # return torch.linalg.lstsq(A, b)[0]


def minimum_norm_in_subspace(
    M: torch.Tensor, K: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Minimize x'Mx subject to Ax=b over x.

    See simulatord.md for a derivation.
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix")
    if K.shape[0] > K.shape[1]:
        raise ValueError("The constraint must be under-determined")

    original_type = M.dtype
    M = M.to(torch.float64)
    K = K.to(torch.float64)
    b = b.to(torch.float64)

    V = solve(M, K.T)
    return (V @ solve(K @ V, b)).to(original_type)


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

    baseline_anchor_width = torch.std(anchor_locations3d).item()

    for anchor_width in tqdm.tqdm(
        torch.linspace(0.05, 0.5, 20) * baseline_anchor_width
    ):
        # A matrix M so that coeffs' M coeffs gives the energy of the field.
        D2 = pairwise_squared_distances(anchor_locations3d, anchor_locations3d)
        M = (
            torch.exp(-D2 / (anchor_width**2 * 4))
            * (1.5 * anchor_width**2 - D2 / 4)
            / anchor_width
        )

        # A matrix that maps coefficients to voltage on the M conductors.
        K = potential_operator(conductor_locations3d, anchor_locations3d, anchor_width)

        coeffs = minimum_norm_in_subspace(M, K, conductor_potentials)

        results.append(
            Result(
                field_energy=(coeffs @ M @ coeffs).item(),
                constraint_violation=torch.abs(K @ coeffs - conductor_potentials)
                .mean()
                .item(),
                potential=Potential(anchor_locations3d, coeffs, anchor_width),
            )
        )

    if verbose:
        anchor_widths = [r.potential.anchor_widths.item() for r in results]

        _, ax = plt.subplots(1, 1)

        # ax.axvline(baseline_anchor_width, ls=":", color="gray")

        best_result = min([r for r in results if r.constraint_violation < 0.01])

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
        print(
            best_result.field_energy,
            best_result.constraint_violation,
            best_result.potential.anchor_widths.item(),
        )

    return best_result.potential
