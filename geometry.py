from typing import NamedTuple, Tuple

import torch


class AxisAlignedInterval(NamedTuple):
    axis: int
    min: float
    max: float


class Face(NamedTuple):
    """A 2D axis-aligned d rectangle that servces as a face of an axis-aligned 3D box.

    The rectangle is a subset of the the axis1-axis2 plane. Its surface normal
    points along offset_axis if orientation=1, or toward the negative
    offset_axis if orientation=-1.  It is offset from the origin by `offset`,
    along the offset_axis dimension.
    """

    first_axis: AxisAlignedInterval
    second_axis: AxisAlignedInterval

    offset_axis: int
    offset: float
    orientation: int


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

    def center(self) -> torch.Tensor:
        return (
            torch.tensor(
                (self.xmin + self.xmax, self.ymin + self.ymax, self.zmin + self.zmax)
            )
            / 2
        )

    def faces(self) -> Tuple[Face]:
        return (
            # The front face.
            Face(
                AxisAlignedInterval(0, self.xmin, self.xmax),
                AxisAlignedInterval(1, self.ymin, self.ymax),
                offset_axis=2,
                offset=self.zmax,
                orientation=+1,
            ),
            # The back face.
            Face(
                AxisAlignedInterval(0, self.xmin, self.xmax),
                AxisAlignedInterval(1, self.ymin, self.ymax),
                offset_axis=2,
                offset=self.zmin,
                orientation=-1,
            ),
            # The top face.
            Face(
                AxisAlignedInterval(0, self.xmin, self.xmax),
                AxisAlignedInterval(2, self.zmin, self.zmax),
                offset_axis=1,
                offset=self.ymax,
                orientation=+1,
            ),
            # The bottom face.
            Face(
                AxisAlignedInterval(0, self.xmin, self.xmax),
                AxisAlignedInterval(2, self.zmin, self.zmax),
                offset_axis=1,
                offset=self.ymin,
                orientation=-1,
            ),
            # The right face.
            Face(
                AxisAlignedInterval(1, self.ymin, self.ymax),
                AxisAlignedInterval(2, self.zmin, self.zmax),
                offset_axis=0,
                offset=self.xmax,
                orientation=+1,
            ),
            # The left face.
            Face(
                AxisAlignedInterval(1, self.ymin, self.ymax),
                AxisAlignedInterval(2, self.zmin, self.zmax),
                offset_axis=0,
                offset=self.xmin,
                orientation=-1,
            ),
        )


def sample_uniform(box: Box, sample_size: int) -> torch.Tensor:
    "Draw points uniformly at random from a box."
    return torch.rand(sample_size, 3) * torch.tensor(box.sizes()) + torch.tensor(
        [box.xmin, box.ymin, box.zmin]
    )


def pairwise_squared_distances(X, Y):
    # D[i,j] = ||x[i]-y[j]||^2
    #        = ||x[i]||^2 + ||y[j]|| - 2x[i]'y[j]
    # So the matrix D is
    #    ||x||^2 + ||y||^2 - 2 x'y
    return (X**2).sum(axis=1)[:, None] + (Y**2).sum(axis=1)[None, :] - 2 * X @ Y.T
