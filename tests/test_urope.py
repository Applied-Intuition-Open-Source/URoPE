# Copyright (c) 2026 Applied Intuition, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from urope.urope import URoPEDotProductAttention

HEADS, HEAD_DIM = 8, 64
CAMERAS, PATCHES_X, PATCHES_Y = 3, 4, 4
WIDTH, HEIGHT = 64, 64


def _random_rotations(n, dtype):
    # Orthonormal to the precision of `dtype` (QR in that dtype), so the
    # SE(3) inverse used inside URoPE is exact up to rounding.
    q, r = torch.linalg.qr(torch.randn(n, 3, 3, dtype=dtype))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[..., None, :]
    q[torch.det(q) < 0, :, 0] *= -1
    return q


def _random_inputs(dtype, batch=2):
    seqlen = CAMERAS * PATCHES_X * PATCHES_Y
    q = torch.randn(batch, HEADS, seqlen, HEAD_DIM, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    c2w = torch.eye(4, dtype=dtype).repeat(batch, CAMERAS, 1, 1)
    c2w[..., :3, :3] = _random_rotations(batch * CAMERAS, dtype).reshape(
        batch, CAMERAS, 3, 3
    )
    c2w[..., :3, 3] = torch.randn(batch, CAMERAS, 3, dtype=dtype)
    Ks = torch.tensor(
        [[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]], dtype=dtype
    ).repeat(batch, CAMERAS, 1, 1)
    return q, k, v, c2w, Ks


def _attention():
    return URoPEDotProductAttention(
        HEADS, HEAD_DIM, PATCHES_X, PATCHES_Y, WIDTH, HEIGHT
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_matches_input_dtype_and_shape(dtype):
    torch.manual_seed(0)
    q, k, v, c2w, Ks = _random_inputs(dtype)
    out = _attention()(q, k, v, viewmats=torch.inverse(c2w), Ks=Ks)
    assert out.shape == q.shape
    assert out.dtype == dtype
    assert torch.isfinite(out).all()


def test_invariant_to_global_rigid_transform():
    # Moving every camera by the same SE(3) transform should not change the
    # output, since only relative camera poses enter the encoding.
    torch.manual_seed(0)
    q, k, v, c2w, Ks = _random_inputs(torch.float64)
    world = torch.eye(4, dtype=torch.float64)
    world[:3, :3] = _random_rotations(1, torch.float64)[0]
    world[:3, 3] = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)

    attn = _attention()
    out = attn(q, k, v, viewmats=torch.inverse(c2w), Ks=Ks)
    out_moved = attn(q, k, v, viewmats=torch.inverse(world @ c2w), Ks=Ks)
    torch.testing.assert_close(out, out_moved)
