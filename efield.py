from typing import NamedTuple, Tuple

import math

import matplotlib.pylab as plt
import torch
from torch import nn
import tqdm

import geometry


def integral_of_gaussian(mean: float, std: float, xmin: float, xmax: float) -> float:
    """Integrate a Gaussian with given mean and variance over the given interval.

    Compute
       int_xmin^xmax exp(-0.5 * (x-mean)^2 / std^2) dx

    Notably, the Gaussian is not normalized.
    """
    # Pytorch's ndtr function computes
    #   1/sqrt(2pi) int_-infty^x exp(-0.5 t^2/2) dt

    # With the change of variables t(x)=(x-mean)/std, the integral we wanted to compute
    # becomes
    #   int_xmin^xmax exp(-(x-mean)^2/std^2/2) dx
    # = int_t(xmin)^t(xmax) exp(-t^2/2) * std * dt
    # = std*sqrt(2pi) * (ndtr(t(xmax)) - ndtr(t(xmin)))
    return (
        std
        * math.sqrt(2 * torch.pi)
        * (
            torch.special.ndtr((xmax - mean) / std)
            - torch.special.ndtr((xmin - mean) / std)
        )
    )


class Potential(nn.Module):
    def forward(self, locations3d: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def field(self, locations3d: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def laplacian(self, locations3d: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def flux_through_face(self, face: geometry.Face) -> float:
        raise NotImplementedError

    def enclosed_charge(self, enclosure: geometry.Box) -> float:
        return sum(self.flux_through_face(face) for face in enclosure.faces())


class LinearPotential(Potential):
    """Map a point in three-dimensional space to a scalar potential.

    This is a class of potential functions that are linear in their parmeters.
    """

    def __init__(
        self,
        anchor_locations3d: torch.Tensor,
        anchor_coeffs: torch.Tensor,
        anchor_parameters: torch.Tensor,
    ):
        super().__init__()
        self.anchor_locations3d = nn.Parameter(anchor_locations3d)
        self.anchor_coeffs = nn.Parameter(anchor_coeffs)
        self.anchor_parameters = nn.Parameter(anchor_parameters)

    def forward(self, locations3d: torch.Tensor) -> torch.Tensor:
        return (
            self.potential_operator(
                locations3d, self.anchor_locations3d, self.anchor_parameters
            )
            @ self.anchor_coeffs
        )

    def field(self, locations3d: torch.Tensor) -> torch.Tensor:
        return (
            self.field_operator(
                locations3d, self.anchor_locations3d, self.anchor_parameters
            )
            @ self.anchor_coeffs
        )

    def laplacian(self, locations3d: torch.Tensor) -> torch.Tensor:
        return (
            self.laplacian_operator(
                locations3d, self.anchor_locations3d, self.anchor_parameters
            )
            @ self.anchor_coeffs
        )

    def flux_through_face(self, face: geometry.Face) -> float:
        return (
            self.flux_through_face_operator(
                face, self.anchor_locations3d, self.anchor_parameters
            )
            @ self.anchor_coeffs
        )

    @classmethod
    def sample_from_laplacian(
        cls,
        universe: geometry.Box,
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
            proposal_sample = geometry.sample_uniform(universe, sample_size)

            # The un-normalized probability of each draw under the target distribution. We're
            # guaranteed that these un-normalized target probabilities are between 0 and
            # 1.
            lap = (
                (
                    cls.laplacian_operator(
                        proposal_sample, anchor_locations3d, anchor_width
                    )
                    @ coeffs
                )
                .abs()
                .log()
                .abs()
            )

            prob_under_target = lap / (lap + 1)

            # Keep each draw with the above probability.
            keep_draw = torch.rand(len(prob_under_target)) < prob_under_target

            sample.append(proposal_sample[keep_draw])

        return torch.vstack(sample)


class CoulombPotential(LinearPotential):
    """Potential functions caused by known charges at known locations.

    The potential has the formo

        f(x) = sum_i q_i / ||x-x_i||,

    where x_i is the location of charge i and q_i is the charge.
    """

    @staticmethod
    def potential_operator(
        locations3d: torch.Tensor,
        anchor_locations3d: torch.Tensor,
        anchor_params: torch.Tensor,
    ) -> torch.Tensor:
        D2 = geometry.pairwise_squared_distances(locations3d, anchor_locations3d)
        D2[D2 < 0.1] = 0.1
        return D2**-0.5

    @classmethod
    def field_operator(
        cls,
        locations3d: torch.Tensor,
        anchor_locations3d: torch.Tensor,
        anchor_params: torch.Tensor,
    ) -> torch.Tensor:
        Dx = locations3d[:, 0][:, None] - anchor_locations3d[:, 0][None, :]
        Dy = locations3d[:, 1][:, None] - anchor_locations3d[:, 1][None, :]
        Dz = locations3d[:, 2][:, None] - anchor_locations3d[:, 2][None, :]

        D2 = geometry.pairwise_squared_distances(locations3d, anchor_locations3d)
        D2[D2 < 0.1] = 0.1

        return torch.stack((Dx, Dy, Dz)) * D2**-1.5

    @staticmethod
    def laplacian_operator(
        locations3d: torch.Tensor,
        anchor_locations3d: torch.Tensor,
        anchor_params: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros((len(locations3d), len(anchor_locations3d)))


class RadialBasisFunctionPotential(LinearPotential):
    @staticmethod
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
            * geometry.pairwise_squared_distances(locations3d, anchor_locations3d)
            / anchor_width**2
        )

    @classmethod
    def field_operator(
        cls,
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
        k = cls.potential_operator(locations3d, anchor_locations3d, anchor_width)

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

    @staticmethod
    def laplacian_operator(
        locations3d: torch.Tensor,
        anchor_locations3d: torch.Tensor,
        anchor_width: torch.Tensor,
    ) -> torch.Tensor:
        D2 = geometry.pairwise_squared_distances(locations3d, anchor_locations3d)
        k = torch.exp(-0.5 * D2 / anchor_width**2)
        return 1 / anchor_width**2 * (D2 / anchor_width**2 - 3) * k

    @staticmethod
    def flux_through_face_operator(
        face: geometry.Face,
        anchor_locations3d: torch.Tensor,
        anchor_width: torch.Tensor,
    ) -> float:
        return (
            -(anchor_width**-2)
            * integral_of_gaussian(
                anchor_locations3d[:, face.first_axis.axis],
                anchor_width,
                face.first_axis.min,
                face.first_axis.max,
            )
            * integral_of_gaussian(
                anchor_locations3d[:, face.second_axis.axis],
                anchor_width,
                face.second_axis.min,
                face.second_axis.max,
            )
            * torch.exp(
                -0.5
                * (face.offset - anchor_locations3d[:, face.offset_axis]) ** 2
                / anchor_width**2
            )
            * (face.offset - anchor_locations3d[:, face.offset_axis])
            * face.orientation
        )

    @staticmethod
    def field_energy_operator(
        anchor_locations3d: torch.Tensor,
        anchor_width: torch.Tensor,
    ) -> torch.Tensor:
        D2 = geometry.pairwise_squared_distances(anchor_locations3d, anchor_locations3d)
        return (
            torch.exp(-D2 / anchor_width**2 / 4)
            * (3 / 2 * anchor_width**2 - D2 / 4)
            / anchor_width
            * torch.pi ** (3 / 2)
        )


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


def guess_anchor_width(anchor_locations3d: torch.Tensor) -> float:
    return torch.linalg.norm(
        anchor_locations3d - anchor_locations3d.mean(axis=0), dim=1
    ).mean()


def fit_radial_basis_function_potential(
    universe: geometry.Box,
    conductor_locations3d: torch.Tensor,
    conductor_potentials: torch.Tensor,
    verbose=True,
    num_rounds=70,
    anchor_width_scale=0.5,
) -> RadialBasisFunctionPotential:
    if not (conductor_locations3d.ndim == 2 and conductor_locations3d.shape[1] == 3):
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
    #
    # There is one additional trick, which is not covered in simulation.md:
    # The anchor locations are added iteratively. At each iteration of the loop
    # below, we sample new anchors wherever the Laplacian of the resulting field
    # is high.

    # Add anchor gradually. Start with the conductors and add from there.
    anchor_locations3d = conductor_locations3d

    num_anchors = []
    energies = []
    constraint_violations = []
    extra_anchor_locations3d = torch.empty((0, 3))
    for _ in tqdm.tqdm(range(num_rounds)):
        # Add the anchor locations that were discovered during the previous
        # iteration of this loop.
        anchor_locations3d = torch.vstack(
            (anchor_locations3d, extra_anchor_locations3d)
        )

        anchor_width = guess_anchor_width(anchor_locations3d) * anchor_width_scale

        # A matrix M so that coeffs' M coeffs gives the energy of the field.
        M = RadialBasisFunctionPotential.field_energy_operator(
            anchor_locations3d, anchor_width
        )

        # This matrix positive definite, but numerical issues can cause it to
        # have negative eigenvalues. Add a small multiple of identity to it to
        # prevent this from happening.
        M += 1e-4 * torch.eye(len(anchor_locations3d))

        # M += 1e-2 * RadialBasisFunctionPotential.potential_operator(
        # anchor_locations3d, anchor_locations3d, anchor_width
        # )

        # A matrix that maps coefficients to voltage on the M conductors.
        K = RadialBasisFunctionPotential.potential_operator(
            conductor_locations3d, anchor_locations3d, anchor_width
        )

        coeffs = minimum_norm_in_subspace(M, K, conductor_potentials)

        # Sample some new anchor points. These are added to the set of anchors
        # on the next iteration of the loop.
        extra_anchor_locations3d = RadialBasisFunctionPotential.sample_from_laplacian(
            universe, anchor_locations3d, anchor_width, coeffs
        )
        num_anchors.append(len(anchor_locations3d))
        energies.append(coeffs @ M @ coeffs)
        constraint_violations.append(
            torch.abs(K @ coeffs - conductor_potentials).mean()
        )

    if verbose:
        _, ax = plt.subplots(1, 1)
        ax.plot(num_anchors, energies, color="c", marker=".")
        ax.set_xlabel("Number of anchors")
        ax.set_ylabel("Field energy", color="c")
        ax.tick_params(axis="y", labelcolor="c")

        ax2 = ax.twinx()
        ax2.plot(num_anchors, constraint_violations, color="r")
        ax2.set_ylabel("Constraint violation", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

    return RadialBasisFunctionPotential(
        anchor_locations3d, coeffs, torch.tensor(anchor_width)
    )
