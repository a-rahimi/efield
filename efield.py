from typing import List, NamedTuple, Tuple

import torch
from torch import nn
import tqdm

import geometry


def affine_projection(A, b):
    """
    Produces an operator that maps any x to the nearest solution of

        A u = b

    In other words, this solves the problem

      min_u ||u-x|| s.t. Au=b

    The solution is returned as an affine operator (offset, proj) that maps x to
    u via

       u = proj @ x + offset
    """
    U, s, Vt = torch.linalg.svd(A, full_matrices=False)

    offset = (Vt.T / s) @ U.T @ b
    proj = torch.eye(A.shape[1]) - Vt.T @ Vt
    return offset, proj


class Potential(nn.Module):
    def forward(self, locations3d: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def field(self, locations3d: torch.Tensor) -> torch.Tensor:
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

        if len(anchor_locations3d) != len(anchor_coeffs):
            raise ValueError("Need one coefficient per anchor")

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

<<<<<<< HEAD
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
        sample_size: int = 50,
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
            lap = torch.abs(
                cls.laplacian_operator(
                    proposal_sample, anchor_locations3d, anchor_width
                )
                @ coeffs
            )

            prob_under_target = lap / (lap + 1)

            # Keep each draw with the above probability.
            keep_draw = torch.rand(len(prob_under_target)) < prob_under_target

            sample.append(proposal_sample[keep_draw])

        return torch.vstack(sample)

=======
>>>>>>> 438d1f8 (WIP: new way to compute fields)

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
        eps: torch.Tensor,
    ) -> torch.Tensor:
        D2 = eps + geometry.pairwise_squared_distances(locations3d, anchor_locations3d)
        return 1 / D2.sqrt()

    @staticmethod
    def field_operator(
        locations3d: torch.Tensor,
        anchor_locations3d: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        Dx = locations3d[:, 0][:, None] - anchor_locations3d[:, 0][None, :]
        Dy = locations3d[:, 1][:, None] - anchor_locations3d[:, 1][None, :]
        Dz = locations3d[:, 2][:, None] - anchor_locations3d[:, 2][None, :]

        D2 = eps + geometry.pairwise_squared_distances(locations3d, anchor_locations3d)

        return torch.stack((Dx, Dy, Dz)) * D2**-1.5

    def reduce_field_energy(self, universe: geometry.Box, step: float) -> float:
        """Find an affine subspace of coefficients that satisfies the conductor
        potential constraints.
        """
        # Get an unbiased estimate of the field energy.  The energy of the field
        # is the expected value of the field magnitude throughout the universe.
        # We obtain an unbiased estimate of this energy by computing the field
        # magnitude on a random set of points in the universe.
        Dx, Dy, Dz = self.field(geometry.sample_uniform(universe, 100))
        field_energy = (Dx**2 + Dy**2 + Dz**2).mean()

        # Compute the gradient of the field energy with respect to the
        # parameters and update them.
        self.anchor_locations3d.grad = None
        self.anchor_coeffs.grad = None
        field_energy.backward(inputs=[self.anchor_locations3d, self.anchor_coeffs])

        with torch.no_grad():
            self.anchor_locations3d -= step * self.anchor_locations3d.grad
            self.anchor_coeffs -= step * self.anchor_coeffs.grad

        return field_energy

    def project_coeffs(
        self, conductor_locations3d: torch.Tensor, conductor_potentials: torch.Tensor
    ):
        offset, proj = affine_projection(
            CoulombPotential.potential_operator(
                conductor_locations3d, self.anchor_locations3d, self.anchor_parameters
            ),
            conductor_potentials,
        )
        with torch.no_grad():
            self.anchor_coeffs[:] = offset + proj @ self.anchor_coeffs


class FittingResult(NamedTuple):
    num_anchors: int
    energy: float
    constraint_violation: float


def fit_coulomb_potential(
    universe: geometry.Box,
    conductor_locations3d: torch.Tensor,
    conductor_potentials: torch.Tensor,
    max_anchors=300,
    num_projected_gradient_descent_steps=100,
) -> Tuple[CoulombPotential, List[FittingResult]]:
    if not (conductor_locations3d.ndim == 2 and conductor_locations3d.shape[1] == 3):
        raise ValueError("conductor_locations3d must be Nx3 tensor.")
    if conductor_potentials.ndim != 1:
        raise ValueError("condctuctor_potential must be an N-dimensional vector.")

    # Initially, anchors are centered near (but not on top of) the conductors.
    potential = CoulombPotential(
        anchor_locations3d=geometry.sample_near_points(
            conductor_locations3d,
            weights=torch.ones(len(conductor_locations3d)),
            num_draws=len(conductor_locations3d) * 2,
        ),
        anchor_coeffs=torch.zeros(len(conductor_locations3d) * 2),
        anchor_parameters=torch.tensor(1e-2),
    )
    extra_anchor_locations3d = torch.zeros((0, 3))

    results: List[FittingResult] = []
    progress = tqdm.tqdm(total=max_anchors)
    while len(potential.anchor_locations3d) < max_anchors:
        # Add the extra anchors discovered in the previous iteration of this loop.
        potential = CoulombPotential(
            anchor_locations3d=torch.vstack(
                (potential.anchor_locations3d, extra_anchor_locations3d)
            ),
            anchor_coeffs=torch.hstack(
                (potential.anchor_coeffs, torch.zeros(len(extra_anchor_locations3d)))
            ),
            anchor_parameters=potential.anchor_parameters,
        )

        # Take stochastic gradient steps to reduce the field's energy. This loop modifies
        # both the coefficients and anchors of the potential.
        for _ in range(num_projected_gradient_descent_steps):
            # Reduce the energy of the field, ignoring the boundary conditions.
            field_energy = potential.reduce_field_energy(universe, step=1e-2)

            # Project the coefficients so the field satisfies the boundary conditions.
            potential.project_coeffs(conductor_locations3d, conductor_potentials)

        # Sample new anchor locations based on constraint violations.
        constraint_defects = torch.abs(
            potential(conductor_locations3d) - conductor_potentials
        )

        extra_anchor_locations3d = geometry.sample_near_points(
            conductor_locations3d,
            weights=constraint_defects,
            num_draws=max(10, len(potential.anchor_locations3d) // 10),
        )

        result = FittingResult(
            num_anchors=len(potential.anchor_locations3d),
            energy=field_energy.item(),
            constraint_violation=constraint_defects.mean().item(),
        )
        results.append(result)
        print(result)
        progress.update(n=len(extra_anchor_locations3d))

    return potential, results
