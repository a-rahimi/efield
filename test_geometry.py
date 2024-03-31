import torch

import geometry


def test_pairwise_squared_distances():
    A = torch.rand(3, 2)
    B = torch.rand(4, 2)

    D_expected = torch.cdist(A.unsqueeze(0), B.unsqueeze(0)).squeeze() ** 2
    D_actual = geometry.pairwise_squared_distances(A, B)
    assert torch.allclose(D_expected, D_actual, atol=1e-4), (D_expected, D_actual)


def test_box_expand():
    b = geometry.Box(-1, 1, -2, 2, -3, 3)

    scale = 1.2
    bexp = b.expand(scale)

    assert torch.allclose(bexp.sizes(), b.sizes() * scale)
    assert torch.allclose(bexp.center(), b.center())
