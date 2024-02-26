import dataclasses as dc

import torch
from torch import nn

class Potential(nn.Module):
    def __init__( self, anchor_locations3d: torch.Tensor, anchor_weights: torch.Tensor, anchor_widths: torch.Tensor):
        super().__init__(self)
        self.anchor_locations3d = nn.Parameter(anchor_locations3d)
        self.anchor_weights = nn.Parameter(anchor_weights)
        self.anchor_widths = nn.Parameter(anchor_widths)

    def forward(self, locations3d: torch.Tensor) -> torch.Tensor:
        k = torch.exp(-0.5 * self.anchor_widths* nn.functional.parwise_distance(self.anchor_locations3d, self.anchor_widths)**2)
        return self.anchor_weights @ k

    def enclosed_charge(sphere_center3d: torch.Tensor, sphere_radius: float) -> float:
        raise NotImplementedError()

def find_potential_function(conductor_locations3d: torch.Tensor, conductor_potentials: torch.Tensor, num_iters: int=100,
                potential: Potential=None) -> Potential:
    potential = potential or Potential()
    optimizer = torch.optim.SGD(potential.parameters())

    # Solve the constrained optimization problem
    #
    #   min_f \int_x ||d/dx f(x)||^2
    #   s.t.  f(x) = V(x) when for all x on the conductors.
    #
    # Since the conductors are given to us as a discrete set of points, which we denote by C, we'll denote by f_C the
    # values of on the conductor, and V_C the given potentials on the conductor.
    #
    # Solve this using a barrier method. This problem is equivalent to
    #
    #   min_f max_l L(f, l) 
    #   where L(f,l) == \int_x ||d/dx f(x)||^2 + (f-V)^2 dfx +  l' (V_C - f_C) 
    #
    # We're looking for a saddle point of L. To do this we'll iterate between gradient steps on f and l.


    for it in range(num_iters):
        loss = potential()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return potential
