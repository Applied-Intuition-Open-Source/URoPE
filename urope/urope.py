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

# Portions of this file are adapted from the reference implementation of
# "PRoPE: Projective Positional Encoding for Multiview Transformers". See the
# NOTICE and THIRD_PARTY_LICENSES files for attribution and the applicable
# third-party license terms.

# How to use URoPE (ray / multiview-aware RoPE in this module):
#
# 1. Module wrapper (typical):
#    attn = URoPEDotProductAttention(
#        head_num, head_dim, patches_x, patches_y, image_width, image_height,
#    )
#    out = attn(q, k, v, viewmats=viewmats, Ks=Ks, **sdpa_kwargs)
#
#    Shapes: q,k,v are (batch, num_heads, seqlen, head_dim) with
#    seqlen == cameras * patches_x * patches_y. viewmats: (batch, cameras, 4, 4).
#    Ks: (batch, cameras, 3, 3) intrinsics (required for ray construction).
#
# 2. Functional API (recomputes geometry and RoPE coefficients every call):
#    out = urope_dot_product_attention(
#        q, k, v,
#        viewmats=viewmats, Ks=Ks,
#        patches_x=..., patches_y=..., image_width=..., image_height=...,
#        group_size=..., depth_num=..., max_depth=..., min_depth=...,
#        **sdpa_kwargs,
#    )

from functools import partial
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F


class URoPEDotProductAttention(torch.nn.Module):
    """URoPE attention."""

    def __init__(
        self,
        head_num: int,
        head_dim: int,
        patches_x: int,
        patches_y: int,
        image_width: int,
        image_height: int,
        min_depth: float = 2.0,
        max_depth: float = 20.0,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
        group_size: int = 4,
        leaveout_head: int = 0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height
        
        self.head_num = head_num
        self.leaveout_head = leaveout_head
        self.depth_num = (head_num - leaveout_head) // group_size
        self.group_size = group_size
        
        self.max_depth = max_depth
        self.min_depth = min_depth

    def forward(
        self,
        q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
        Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
        **kwargs,
    ) -> torch.Tensor:
        return urope_dot_product_attention(
            q,
            k,
            v,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            group_size=self.group_size,
            depth_num=self.depth_num,
            max_depth=self.max_depth,
            min_depth=self.min_depth,
            leaveout_head=self.leaveout_head,
            **kwargs,
        )


def urope_dot_product_attention(
    q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    *,
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    group_size: int,
    depth_num: int,
    max_depth: float,
    min_depth: float,
    leaveout_head: int = 0,
    **kwargs,
) -> torch.Tensor:
    """Like F.scaled_dot_product_attention, with URoPE-style ray / image-plane RoPE.

    Assumes self-attention shapes and ``seqlen == cameras * patches_x * patches_y``,
    with tokens ordered so ``seqlen`` can be viewed as ``(cameras, patches_x, patches_y)``.
    """
    # We're going to assume self-attention: all inputs are the same shape.
    (batch, num_heads, seqlen, head_dim) = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
    assert seqlen == cameras * patches_x * patches_y

    apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns(
        head_dim=head_dim,
        viewmats=viewmats,
        Ks=Ks,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
        group_size=group_size,
        depth_num=depth_num,
        max_depth=max_depth,
        min_depth=min_depth,
    )

    patch_num = patches_x * patches_y
    q = q.reshape(batch, num_heads, cameras, patch_num, head_dim)
    q = q.permute(0, 2, 1, 3, 4).reshape(batch*cameras, num_heads, patch_num, head_dim)  # [batch*cameras, num_heads, patch_num, head_dim]
    k = k.reshape(batch, num_heads, seqlen, head_dim)
    k = k[:, None].repeat(1, cameras, 1, 1, 1)
    k = k.flatten(0, 1)  # [batch*cameras, num_heads, seqlen, head_dim]
    v = v.reshape(batch, num_heads, seqlen, head_dim)
    v = v[:, None].repeat(1, cameras, 1, 1, 1)
    v = v.flatten(0, 1)  # [batch*cameras, num_heads, seqlen, head_dim]
    
    if leaveout_head > 0:
        q_1 = q[:, :-leaveout_head]
        q_2 = q[:, -leaveout_head:]
        k_1 = k[:, :-leaveout_head]
        k_2 = k[:, -leaveout_head:]
        v_1 = v[:, :-leaveout_head]
        v_2 = v[:, -leaveout_head:]
        
        q_1 = apply_fn_q(q_1)
        k_1 = apply_fn_kv(k_1)
        v_1 = apply_fn_kv(v_1)
        
        q = torch.cat([q_1, q_2], dim=1)
        k = torch.cat([k_1, k_2], dim=1)
        v = torch.cat([v_1, v_2], dim=1)
    else:
        q = apply_fn_q(q)
        k = apply_fn_kv(k)
        v = apply_fn_kv(v)

    out = F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        **kwargs,
    )
    if leaveout_head > 0:
        out_1 = out[:, :-leaveout_head]
        out_2 = out[:, -leaveout_head:]
        out_1 = apply_fn_o(out_1)
        out = torch.cat([out_1, out_2], dim=1)
    else:
        out = apply_fn_o(out)  # [batch*cameras, num_heads, patch_num, head_dim]
    out = out.reshape(batch, cameras, num_heads, patch_num, head_dim)
    out = out.permute(0, 2, 1, 3, 4).reshape(batch, num_heads, seqlen, head_dim)
    assert out.shape == (batch, num_heads, seqlen, head_dim)

    return out


def _prepare_apply_fns(
    head_dim: int,  # Q/K/V will have this last dimension
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    group_size: int,
    depth_num: int,
    max_depth: float,
    min_depth: float,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare transforms for URoPE-style positional encoding."""
    # viewmats is `camera<-world` transform (x-right, y-up, z-backward).

    device = viewmats.device
    coordinate_transforms = torch.tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]).to(device)
    viewmats = coordinate_transforms[None, None] @ viewmats # (batch, cameras, 4, 4)
    num_cameras = viewmats.shape[1]
    # now viewmats is `camera<-world` transform (x-right, y-down, z-forward).
    
    (batch, cameras, _, _) = viewmats.shape
    Ks_inv = _invert_K(Ks)  # (batch, cameras, 3, 3)
    viewmats_cam2world = _invert_SE3(viewmats)  # (batch, cameras, 4, 4)

    grid_x, grid_y = torch.meshgrid(torch.arange(patches_x), torch.arange(patches_y), indexing="xy")
    img_x = (grid_x + 0.5) * (image_width / patches_x)
    img_y = (grid_y + 0.5) * (image_height / patches_y)
    img_z = torch.ones_like(img_x)
    img_xyz = torch.stack([img_x, img_y, img_z], dim=-1).to(device)  # (patch_y, patch_x, 3)
    img_xyz = img_xyz.reshape(-1, 3)[None, None].repeat(batch, cameras, 1, 1)  # (batch, cameras, patch_num, 3)
    cam_xyz = img_xyz @ Ks_inv.transpose(-1, -2) # (batch, cameras, patch_num, 3)
    cam_xyz_pad = torch.cat([cam_xyz, torch.ones_like(cam_xyz[..., :1])], dim=-1) # (batch, cameras, patch_num, 4)
    
    # get the ray map in the camera coordinate
    raymap = cam_xyz_pad @ viewmats_cam2world.transpose(-1, -2)  # (batch, cameras, patch_num, 4)
    raydir = raymap[..., :3] - viewmats_cam2world[..., None, :3, 3]

    P = viewmats_cam2world[:, :, None].repeat(1, 1, patches_x * patches_y, 1, 1)

    cam_centers = P[..., :3, 3]  # (batch, cameras, patch_num, 3) world coordinates
    ray_ends = cam_centers + raydir  # (batch, cameras, patch_num, 3) world coordinates
    
    cam2world = viewmats_cam2world   # (batch, cameras, 4, 4)
    
    cam2cam = _invert_SE3(cam2world)[:, None] @ cam2world[:, :, None]  # (batch, cameras, cameras, 4, 4)  1 --> 2

    cam_centers_pad = torch.cat([cam_centers, torch.ones_like(cam_centers[..., :1])], dim=-1)  # (batch, cameras, patch_num, 4)
    ray_ends_pad = torch.cat([ray_ends, torch.ones_like(ray_ends[..., :1])], dim=-1)  # (batch, cameras, patch_num, 4)
    
    cam_centers_pad = cam_centers_pad @ viewmats.transpose(-1, -2) # (batch, cameras, patch_num, 4) camera coordinates
    ray_ends_pad = ray_ends_pad @ viewmats.transpose(-1, -2) # (batch, cameras, patch_num, 4) camera coordinates
    
    # convert the ray map into the query camera coordinate
    cam_centers_camcoord = cam_centers_pad[:, :, None] @ cam2cam.transpose(-1, -2)  # (batch, cameras, cameras, patch_num, 4)  each cam_1 in cam_2 coordinates
    ray_ends_camcoord = ray_ends_pad[:, :, None] @ cam2cam.transpose(-1, -2)  # (batch, cameras, cameras, patch_num, 4)
    
    cam_centers_camcoord = cam_centers_camcoord[..., :3]  # (batch, cameras, cameras, patch_num, 3)
    ray_ends_camcoord = ray_ends_camcoord[..., :3]  # (batch, cameras, cameras, patch_num, 3)
    ray_dirs_camcoord = ray_ends_camcoord - cam_centers_camcoord  # (batch, cameras, cameras, patch_num, 3)

    cam_ids = torch.arange(cameras, device=device)[None, :, None, None, None].repeat(
        batch, 1, 1, patches_x * patches_y, 3
    )  # [batch, cameras, 1, patch_num, 3]
    q_cam_centers_camcoord = torch.gather(cam_centers_camcoord, dim=2, index=cam_ids)  # [batch, cameras, 1, patch_num, 3]
    q_ray_dirs_camcoord = torch.gather(ray_dirs_camcoord, dim=2, index=cam_ids)  # [batch, cameras, 1, patch_num, 3]
    
    q_cam_centers_camcoord = q_cam_centers_camcoord.squeeze(2)  # [batch, cameras, patch_num, 3]
    q_ray_dirs_camcoord = q_ray_dirs_camcoord.squeeze(2)  # [batch, cameras, patch_num, 3]
    
    
    q_ray_x = q_ray_dirs_camcoord[..., 0].flatten(0, 1)  # [batch*cameras, patch_num]
    q_ray_y = q_ray_dirs_camcoord[..., 1].flatten(0, 1)  # [batch*cameras, patch_num]
    q_ray_z = q_ray_dirs_camcoord[..., 2].flatten(0, 1)  # [batch*cameras, patch_num]
    
    q_cam_x = q_cam_centers_camcoord[..., 0].flatten(0, 1)  # [batch*cameras, patch_num]
    q_cam_y = q_cam_centers_camcoord[..., 1].flatten(0, 1)  # [batch*cameras, patch_num]
    q_cam_z = q_cam_centers_camcoord[..., 2].flatten(0, 1)  # [batch*cameras, patch_num]
    
    
    # sample the depth anchors along the ray
    index = torch.arange(start=0, end=depth_num, step=1, device=device).float()
    bin_size = (max_depth - min_depth) / depth_num
    depths = min_depth + bin_size * index
    depths = depths[None, :, None]  # [1, depth_num, 1]
    
    q_x = q_cam_x[:, None] + q_ray_x[:, None] * depths  # [batch*cameras, depth_num, patch_num]
    q_y = q_cam_y[:, None] + q_ray_y[:, None] * depths  # [batch*cameras, depth_num, patch_num]
    q_z = q_cam_z[:, None] + q_ray_z[:, None] * depths  # [batch*cameras, depth_num, patch_num]
    
    q_x = q_x.repeat(1, group_size, 1)  # [batch*cameras, num_heads, patch_num]
    q_y = q_y.repeat(1, group_size, 1)  # [batch*cameras, num_heads, patch_num]
    q_z = q_z.repeat(1, group_size, 1)  # [batch*cameras, num_heads, patch_num]
    
    q_img_x = img_x / (image_width / patches_x)
    q_img_y = img_y / (image_height / patches_y)
    q_img_x = q_img_x.flatten(0, 1)[None, None].repeat(batch*cameras, depth_num*group_size, 1)  # [batch*cameras, num_heads, patch_num]
    q_img_y = q_img_y.flatten(0, 1)[None, None].repeat(batch*cameras, depth_num*group_size, 1)  # [batch*cameras, num_heads, patch_num]
    q_img_x = q_img_x.type_as(q_x)
    q_img_y = q_img_y.type_as(q_y)

    q_coeffs_img_x = _rope_precompute_coeffs_batch(q_img_x, 100, 1, head_dim // 2) 
    q_coeffs_img_y = _rope_precompute_coeffs_batch(q_img_y, 100, 1, head_dim // 2)
    

    cam_centers_camcoord = cam_centers_camcoord.permute(0, 2, 1, 3, 4)
    ray_dirs_camcoord = ray_dirs_camcoord.permute(0, 2, 1, 3, 4)
    kv_cam_x = cam_centers_camcoord[..., 0].reshape(batch*num_cameras, num_cameras*patches_x * patches_y) # [batch*cameras, cameras*patch_num]
    kv_cam_y = cam_centers_camcoord[..., 1].reshape(batch*num_cameras, num_cameras*patches_x * patches_y) # [batch*cameras, cameras*patch_num]
    kv_cam_z = cam_centers_camcoord[..., 2].reshape(batch*num_cameras, num_cameras*patches_x * patches_y) # [batch*cameras, cameras*patch_num]
    
    kv_ray_x = ray_dirs_camcoord[..., 0].reshape(batch*num_cameras, num_cameras*patches_x * patches_y) # [batch*cameras, cameras*patch_num]
    kv_ray_y = ray_dirs_camcoord[..., 1].reshape(batch*num_cameras, num_cameras*patches_x * patches_y) # [batch*cameras, cameras*patch_num]
    kv_ray_z = ray_dirs_camcoord[..., 2].reshape(batch*num_cameras, num_cameras*patches_x * patches_y) # [batch*cameras, cameras*patch_num]
    
    
    kv_x = kv_cam_x[:, None] + kv_ray_x[:, None] * depths  # [batch*cameras, depth_num, cameras*patch_num]
    kv_y = kv_cam_y[:, None] + kv_ray_y[:, None] * depths  # [batch*cameras, depth_num, cameras*patch_num]
    kv_z = kv_cam_z[:, None] + kv_ray_z[:, None] * depths  # [batch*cameras, depth_num, cameras*patch_num]
    
    kv_xyz = torch.stack([kv_x, kv_y, kv_z], dim=-1) # [batch*cameras, depth_num, cameras*patch_num, 3] 
    kv_img_xyz = kv_xyz @ Ks.flatten(0, 1)[:, None].transpose(-1, -2) # [batch*cameras, depth_num, cameras*patch_num, 3] 
    
    kv_img_x = kv_img_xyz[..., 0] / (torch.abs(kv_img_xyz[..., 2]) + 1e-5) # [batch*cameras, depth_num, cameras*patch_num]
    kv_img_y = kv_img_xyz[..., 1] / (torch.abs(kv_img_xyz[..., 2]) + 1e-5) # [batch*cameras, depth_num, cameras*patch_num]
    
    kv_img_x = kv_img_x / (image_width / patches_x)
    kv_img_y = kv_img_y / (image_height / patches_y)
    
    kv_img_x = torch.clamp(kv_img_x, min=-64, max=96)
    kv_img_y = torch.clamp(kv_img_y, min=-64, max=96)
    
    kv_img_x = kv_img_x.repeat_interleave(group_size, dim=1)  # [batch*cameras, num_heads, cameras*patch_num]
    kv_img_y = kv_img_y.repeat_interleave(group_size, dim=1)  # [batch*cameras, num_heads, cameras*patch_num]

    kv_coeffs_img_x = _rope_precompute_coeffs_batch(kv_img_x, 100, 1, head_dim // 2)
    kv_coeffs_img_y = _rope_precompute_coeffs_batch(kv_img_y, 100, 1, head_dim // 2)

    transforms_q = [
        (partial(_rope_apply_coeffs, coeffs=q_coeffs_img_x), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=q_coeffs_img_y), head_dim // 2),
    ]
    transforms_kv = [
        (partial(_rope_apply_coeffs, coeffs=kv_coeffs_img_x), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=kv_coeffs_img_y), head_dim // 2),
    ]
    transforms_o = [
        (partial(_rope_apply_coeffs, coeffs=q_coeffs_img_x, inverse=True), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=q_coeffs_img_y, inverse=True), head_dim // 2),
    ]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_fn_q, apply_fn_kv, apply_fn_o


def _rope_precompute_coeffs_batch(
    positions: torch.Tensor,  # (batch, num_heads, seqlen)
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE coefficients."""
    assert len(positions.shape) == 3
    assert feat_dim % 2 == 0
    num_heads = positions.shape[1]
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base
        ** (
            -torch.arange(num_freqs, device=positions.device)[None, None, None, :]
            / num_freqs
        )
    )
    angles = positions[:, :, :, None] * freqs
    # Shape should be: `(batch, num_heads, seqlen, num_freqs)`; we're
    # broadcasting across `batch` and `num_heads`.
    assert angles.shape == (positions.shape[0], num_heads, positions.shape[2], num_freqs)
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE coefficients to features. We adopt a 'split' ordering
    convention. (in contrast to 'interleaved')"""
    cos, sin = coeffs
    # We allow (cos, sin) to be either with shape (1, 1, seqlen, feat_dim // 2),
    # or (1, 1, seqlen_per_image, feat_dim // 2) and we repeat it to
    # match the shape of feats.
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    assert len(feats.shape) == len(cos.shape) == len(sin.shape) == 4
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    return torch.cat(
        (
            [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
            if not inverse
            else [cos * x_in - sin * y_in, sin * x_in + cos * y_in]
        ),
        dim=-1,
    )


def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Tensor) -> Tensor, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )

    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out
