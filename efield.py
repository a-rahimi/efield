from typing import NamedTuple, Tuple

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

    def sizes(self) -> Tuple[float, float, float]:
        return (
            (self.xmax - self.xmin),
            (self.ymax - self.ymin),
            (self.zmax - self.zmin),
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


def laplacian_operator(
    locations3d: torch.Tensor,
    anchor_locations3d: torch.Tensor,
    anchor_width: torch.Tensor,
) -> torch.Tensor:
    D2 = pairwise_squared_distances(locations3d, anchor_locations3d)
    k = torch.exp(-0.5 * D2 / anchor_width**2)
    return 1 / anchor_width**2 * (D2 / anchor_width**2 - 3) * k


def sample_uniform(box: Box, sample_size: int) -> torch.Tensor:
    return torch.rand(sample_size, 3) * torch.tensor(box.sizes()) + torch.tensor(
        [box.xmin, box.ymin, box.zmin]
    )


def sample_from_laplacian(
    universe: Box,
    anchor_locations3d: torch.Tensor,
    anchor_width: torch.Tensor,
    coeffs: torch.Tensor,
    sample_size: int = 10,
):
    # We want to draw a sample from a distribution p(x). We don't know how to
    # directly sample from p(x), and we can only evaluate it up to a constant.
    # In other words, our only access to p(x) is through a function psi(x) so
    # that p(x) = 1/Z psi(x), and Z is unknown. Our approach is to first draw
    # instead a sample from a uniform proposal distribution q(x). We will keep
    # each draw with probability psi(x).
    #
    # Here is how to show that this results in a sample from p: We can use it to
    # compute the expectation of any function f(x) under p(x).  Let S denote the
    # sample obtained from the above scheme. Then
    #
    #   1/(V Z n) sum_{x in S} f(x)
    #
    # with x_i ~ q is an unbiased estimate of E f(x) with x~p. Because this
    # holds for all f, this implies that S is a sample from p.
    #
    # To show that the above estimator is an unbiased estimate of E_{x~p} f(x),
    # let the boolean w_i denote whether we accepted sample x_i, so that E[w_i |
    # x_i] = psi(x_i). Then the estimator boils down to
    #
    #   1/(V Z n) sum_i w_i f(x_i).
    #
    # Taking expectations with respect to w (conditioned on x) gives
    #
    #   1/(V Z n) sum_i psi(x_i) f(x_i).
    #
    # Then taking expectation over x~q gives
    #
    #   1/n sum_i \int psi(x)/Z f(x) dx = E_{x~p) f(x).

    sample = []
    while len(sample) < sample_size:
        # Draw proposals from a uniform distribution.
        proposal_sample = sample_uniform(universe, sample_size)

        # The un-normalized probability of each draw under the target distribution. We're
        # guaranteed that these un-normalized target probabilities are between 0 and
        # 1.
        prob_under_target = (
            2
            * torch.special.expit(
                torch.abs(
                    laplacian_operator(
                        proposal_sample, anchor_locations3d, anchor_width
                    )
                    @ coeffs
                )
            )
            - 1
        )

        # Keep each draw with the above probability.
        keep_draw = torch.rand(len(prob_under_target)) < prob_under_target

        sample.append(proposal_sample[keep_draw])

    return torch.vstack(sample)


class Potential(nn.Module):
    """Map a point in three-dimensional space to a scalar potential."""

    def __init__(
        self,
        anchor_locations3d: torch.Tensor,
        anchor_coeffs: torch.Tensor,
        anchor_widths: torch.Tensor,
    ):
        super().__init__()
        self.anchor_locations3d = nn.Parameter(anchor_locations3d)
        self.anchor_coeffs = nn.Parameter(anchor_coeffs)
        self.anchor_widths = nn.Parameter(anchor_widths)

    def forward(self, locations3d: torch.Tensor) -> torch.Tensor:
        return (
            potential_operator(locations3d, self.anchor_locations3d, self.anchor_widths)
            @ self.anchor_coeffs
        )

    def field(self, locations3d: torch.Tensor) -> torch.Tensor:
        return (
            field_operator(locations3d, self.anchor_locations3d, self.anchor_widths)
            @ self.anchor_coeffs
        )

    def laplacian(self, locations3d: torch.Tensor) -> torch.Tensor:
        return (
            laplacian_operator(locations3d, self.anchor_locations3d, self.anchor_widths)
            @ self.anchor_coeffs
        )

    def enclosed_charge(enclosure: Box) -> float:
        raise NotImplementedError()


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

    V = torch.linalg.solve(M, K.T)
    return (V @ torch.linalg.solve(K @ V, b)).to(original_type)


def potential_function(
    universe: Box,
    conductor_locations3d: torch.Tensor,
    conductor_potentials: torch.Tensor,
    verbose=True,
) -> Potential:
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

    anchor_locations3d = conductor_locations3d

    num_anchors = []
    energies = []
    constraint_violations = []
    extra_anchor_locations3d = None
    for _ in tqdm.tqdm(range(100)):
        if extra_anchor_locations3d is not None:
            anchor_locations3d = torch.vstack(
                (anchor_locations3d, extra_anchor_locations3d)
            )

        anchor_width = torch.std(anchor_locations3d).item() * 1.0
        print("anchor width:", anchor_width)

        # A matrix that stores the squared distance between every pair of anchor locations.
        D2 = pairwise_squared_distances(anchor_locations3d, anchor_locations3d)

        # A matrix M so that coeffs' M coeffs gives the energy of the field.
        M = (
            torch.exp(-D2 / (anchor_width**2 * 4))
            * (1.5 * anchor_width**2 - D2 / 4)
            / anchor_width
        )
        M += 1e-4 * torch.eye(len(anchor_locations3d))

        # A matrix that maps coefficients to voltage on the M conductors.
        K = potential_operator(conductor_locations3d, anchor_locations3d, anchor_width)

        coeffs = minimum_norm_in_subspace(M, K, conductor_potentials)

        # Sample some new anchor points. These are added to the set of anchors
        # on the next iteration of the loop.
        extra_anchor_locations3d = sample_from_laplacian(
            universe, anchor_locations3d, anchor_width, coeffs
        )
        num_anchors.append(len(anchor_locations3d))
        energies.append(coeffs @ M @ coeffs)
        constraint_violations.append(
            torch.abs(K @ coeffs - conductor_potentials).mean()
        )

    if verbose:
        _, ax = plt.subplots(1, 1)
        ax.plot(num_anchors, energies, color="c")
        ax.set_xlabel("Number of anchors")
        ax.set_ylabel("Field energy", color="c")
        ax.tick_params(axis="y", labelcolor="c")

        ax2 = ax.twinx()
        ax2.plot(num_anchors, constraint_violations, color="r")
        ax2.set_ylabel("Constraint violation", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

    return Potential(anchor_locations3d, coeffs, torch.tensor(anchor_width))
