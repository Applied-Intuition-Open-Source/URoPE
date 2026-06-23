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

import glob
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _normalize_poses(
    in_c2ws: torch.Tensor,
    scene_scale_factor: float = 1.35,
):
    """
    From: https://github.com/Haian-Jin/LVSM/blob/ebeff4989a3e1ec38fcd51ae24919d0eadf38c8f/data/dataset_scene.py#L54-L95
    Preprocess the poses to:
    1. translate and rotate the scene to align the average camera direction and position
    2. rescale the whole scene to a fixed scale
    """

    # Translation and Rotation
    # align coordinate system (OpenCV coordinate) to the mean camera
    # center is the average of all camera centers
    # average direction vectors are computed from all camera direction vectors (average down and forward)
    center = in_c2ws[:, :3, 3].mean(0)
    avg_forward = F.normalize(
        in_c2ws[:, :3, 2].mean(0), dim=-1
    )  # average forward direction (z of opencv camera)
    avg_down = in_c2ws[:, :3, 1].mean(0)  # average down direction (y of opencv camera)
    avg_right = F.normalize(
        torch.cross(avg_down, avg_forward, dim=-1), dim=-1
    )  # (x of opencv camera)
    avg_down = F.normalize(
        torch.cross(avg_forward, avg_right, dim=-1), dim=-1
    )  # (y of opencv camera)

    avg_pose = torch.eye(4, device=in_c2ws.device)  # average c2w matrix
    avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
    avg_pose[:3, 3] = center
    avg_pose = torch.linalg.inv(avg_pose)  # average w2c matrix
    in_c2ws = avg_pose @ in_c2ws

    # Rescale the whole scene to a fixed scale
    scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
    scene_scale = scene_scale_factor * scene_scale

    in_c2ws[:, :3, 3] /= scene_scale
    return in_c2ws


def _normalize_poses_identity_unit_distance(
    in_c2ws: torch.Tensor,
    ref0_idx: int,
    ref1_idx: int,
):
    """
    Normalize the poses such that the ref0 camera is the identity
    and the ref1 camera is unit distance to the ref0 camera.
    """

    ref0_c2w = in_c2ws[ref0_idx]
    c2ws = torch.einsum("ij,njk->nik", torch.linalg.inv(ref0_c2w), in_c2ws)

    # ref1_c2w = c2ws[ref1_idx]
    # # dist = torch.linalg.norm(ref1_c2w[:3, 3] - ref0_c2w[:3, 3])
    # dist = torch.linalg.norm(ref1_c2w[:3, 3])
    # if dist > 1e-2:  # numerically stable
    #     c2ws[:, :3, 3] /= dist
    

    return c2ws


def resize_crop_with_subpixel_accuracy(
    image: np.ndarray, K: np.ndarray, patch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize and crop the image to have the smallest side equal to `patch_size`,
    while maintaining sub-pixel accuracy using a single warpAffine transformation.

    Args:
        image (np.ndarray): Input image.
        K (np.ndarray): Camera intrinsic matrix.
        patch_size (int): Target size of the smaller dimension.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Resized and cropped image, updated intrinsic matrix.
    """
    h, w = image.shape[:2]
    scale = patch_size / min(h, w)

    # Compute the affine transformation matrix combining scaling and cropping
    new_w, new_h = w * scale, h * scale
    crop_x = (new_w - patch_size) / 2
    crop_y = (new_h - patch_size) / 2

    M = np.array([[scale, 0, -crop_x], [0, scale, -crop_y]], dtype=np.float32)

    # Apply affine transformation with sub-pixel accuracy
    is_downsampling = min(h, w) > patch_size
    interpolation = cv2.INTER_AREA if is_downsampling else cv2.INTER_CUBIC
    cropped_resized_image = cv2.warpAffine(
        image, M, (patch_size, patch_size), flags=interpolation
    )

    # Update intrinsic matrix K
    K_scaled = K.copy()
    K_scaled[:2, :] *= scale
    K_scaled[0, 2] -= crop_x
    K_scaled[1, 2] -= crop_y

    return cropped_resized_image, K_scaled


def center_zoom_in_with_subpixel_accuracy(
    image: np.ndarray, K: np.ndarray, scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Zoom into the center of the image while maintaining sub-pixel accuracy using warpAffine.

    Args:
        image (np.ndarray): Input image.
        K (np.ndarray): Camera intrinsic matrix.
        scale (float): Zoom-in factor.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Zoomed image, updated intrinsic matrix.
    """
    if scale == 1.0:
        return image, K

    h, w = image.shape[:2]
    center_x, center_y = w / 2, h / 2

    # Compute the affine transformation matrix for zooming in at the center
    M = np.array(
        [[scale, 0, (1 - scale) * center_x], [0, scale, (1 - scale) * center_y]],
        dtype=np.float32,
    )

    # Apply affine transformation
    zoomed_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_AREA)

    # Update intrinsic matrix K.
    K_zoomed = K.copy()
    K_zoomed[:2, :] *= scale
    K_zoomed[0, 2] += (1 - scale) * center_x
    K_zoomed[1, 2] += (1 - scale) * center_y

    return zoomed_image, K_zoomed


def load_and_maybe_update_meta_info(data_dir, data_info) -> Tuple[bool, Dict]:
    """Load the meta information for each scene.
    meta_info = {
        "frames": List[Dict],
        "w": int,
        "h": int,
        "fl_x": float,
        "fl_y": float,
        "cx": float,
        "cy": float,
    }
    """
    
    context_views = data_info["context_views"]
    target_views = data_info["target_views"]
    context_views_scales = data_info["context_views_scales"]
    target_views_scales = data_info["target_views_scales"]
    frames = []
    for view_id, scale in zip(context_views, context_views_scales):
        transform_matrix = np.load(os.path.join(data_dir, view_id + ".npy"))
        transform_matrix_pad = np.eye(4)
        transform_matrix_pad[:3] = transform_matrix
        transform_matrix_pad = np.linalg.inv(transform_matrix_pad)
        frame = {
            "file_path": os.path.join(data_dir, view_id + ".png"),
            "transform_matrix": transform_matrix_pad,
            "scale": scale,
        }
        frames.append(frame)
    for view_id, scale in zip(target_views, target_views_scales):
        transform_matrix = np.load(os.path.join(data_dir, view_id + ".npy"))
        transform_matrix_pad = np.eye(4)
        transform_matrix_pad[:3] = transform_matrix
        transform_matrix_pad = np.linalg.inv(transform_matrix_pad)
        frame = {
            "file_path": os.path.join(data_dir, view_id + ".png"),
            "transform_matrix": transform_matrix_pad,
            "scale": scale,
        }
        frames.append(frame)
    
    meta_info = {
        "frames": frames,
        "w": 256,
        "h": 256,
    }
    meta_info["cx"] = meta_info["w"] / 2
    meta_info["cy"] = meta_info["h"] / 2
    hfov = np.deg2rad(49.1)
    meta_info["fl_x"] = meta_info["w"] / (2 * np.tan(hfov / 2))
    meta_info["fl_y"] = meta_info["fl_x"]

    return True, meta_info


def random_intrinsic_perturbation(K: np.ndarray, img: np.ndarray, scale: float = None, scale_range=(0.4, 1.6)) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly change the focal length."""
    if scale is None:
        scale = np.random.uniform(scale_range[0], scale_range[1])
    H, W = img.shape[:2]

    # 1) resize
    newW = int(round(W * scale))
    newH = int(round(H * scale))
    resized = cv2.resize(img, (newW, newH))

    # 2) center crop/pad to original size
    out = None
    if scale >= 1.0:
        # center crop
        x0 = (newW - W) // 2
        y0 = (newH - H) // 2
        out = resized[y0:y0 + H, x0:x0 + W].copy()

        # 3) update K: first scale, then crop shift
        # pixel mapping: u_resized = scale*u, v_resized = scale*v
        # then crop: u_out = u_resized - x0, v_out = v_resized - y0
        S = np.array([[scale, 0, 0],
                      [0, scale, 0],
                      [0,    0, 1]], dtype=np.float64)
        T = np.array([[1, 0, -x0],
                      [0, 1, -y0],
                      [0, 0,   1]], dtype=np.float64)

        K_new = T @ (S @ K)

    else:
        # center pad (black)
        # place resized into black canvas of original size
        canvas_shape = (H, W) if img.ndim == 2 else (H, W, img.shape[2])
        out = np.zeros(canvas_shape, dtype=img.dtype)

        x0 = (W - newW) // 2
        y0 = (H - newH) // 2
        out[y0:y0 + newH, x0:x0 + newW] = resized

        # 3) update K: first scale, then pad shift
        # u_resized = scale*u, v_resized = scale*v
        # then pad: u_out = u_resized + x0, v_out = v_resized + y0
        S = np.array([[scale, 0, 0],
                      [0, scale, 0],
                      [0,    0, 1]], dtype=np.float64)
        T = np.array([[1, 0, x0],
                      [0, 1, y0],
                      [0, 0,  1]], dtype=np.float64)

        K_new = T @ (S @ K)

    return K_new, out



def load_frames_from_meta_info(
    data_dir: str,
    meta_info: Dict[str, Any],
    frame_ids: List[int],
    patch_size: int = 256,
    zoom_factor: float = 1.0,
    random_zoom: bool = False,
    camera_pose_only: bool = False,
    random_intrinsic: bool = False,
    random_scale: bool = False,
) -> Union[Dict[str, Any], np.ndarray]:

    # Load the camera intrinsic
    K_raw = np.array(
        [
            [meta_info["fl_x"], 0, meta_info["cx"]],
            [0, meta_info["fl_y"], meta_info["cy"]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    frames = meta_info["frames"]

    global_scale_factor = np.random.uniform(0.5, 2.0)

    # Shortcut for loading only camera poses
    if camera_pose_only:
        c2ws = []
        for frame_id in frame_ids:
            frame = frames[frame_id]
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)
            if random_scale:
                c2w[:3, 3] *= global_scale_factor
            c2ws.append(c2w)
        return np.stack(c2ws)

    # Load the images
    images, Ks, c2ws, abs_image_paths = [], [], [], []

    for index, frame_id in enumerate(frame_ids):
        frame = frames[frame_id]

        abs_image_path = frame["file_path"]
        scale = frame["scale"]
        try:
            image = imageio.imread(abs_image_path)[..., :3]
            image = cv2.resize(image, (meta_info["w"], meta_info["h"]))
        except Exception as e:
            print(f"Error loading image {abs_image_path}: {e}")
            continue
        
        per_image_zoom_factor = (
            np.random.uniform(1.0, zoom_factor) if random_zoom else zoom_factor
        )

        image, K = center_zoom_in_with_subpixel_accuracy(
            image,
            K_raw,
            per_image_zoom_factor,
        )
        image, K = resize_crop_with_subpixel_accuracy(image, K, patch_size)

        if random_intrinsic:
            K, image = random_intrinsic_perturbation(K, image)
        else:
            K, image = random_intrinsic_perturbation(K, image, scale=scale)

        c2w = np.array(frame["transform_matrix"], dtype=np.float32)

        if random_scale:
            c2w[:3, 3] *= global_scale_factor

        images.append(image)
        Ks.append(K)
        c2ws.append(c2w)
        abs_image_paths.append(abs_image_path)

    return {
        "image": np.stack(images),
        "K": np.stack(Ks),
        "camtoworld": np.stack(c2ws),
        "image_path": abs_image_paths,
    }


class ObjaverseTrainDataset(Dataset):
    def __init__(
        self,
        train_index_file: str,
        patch_size: int = 256,
        zoom_factor: float = 1.0,  # 1.0 means disabled
        random_zoom: bool = False,  # only useful when zoom_factor is > 1.0
        input_views: int = 2,
        supervise_views: int = 6,
        verbose: bool = False,
        dataset_dir: str = "data_processed/objaverse/views_release/",
    ):
        super().__init__()
        # No list/dict in the dataset, which would cause "memory leak"
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-715050814
        self.dataset_dir = dataset_dir
        with open(train_index_file, "r") as f:
            self.train_index_data = json.load(f)
        self.train_ids = list(self.train_index_data.keys())
        self.patch_size = patch_size
        self.zoom_factor = zoom_factor
        self.random_zoom = random_zoom
        self.input_views = input_views
        self.supervise_views = supervise_views
        self.verbose = verbose
        if self.verbose:
            print(f"[TrainDataset] Initialized with {len(self.train_index_data)} scenes.")

    def __len__(self):
        return len(self.train_index_data)

    def _select_views(
        self, n_frames: int, min_frame_dist: int = 25, max_frame_dist: int = 100
    ) -> Optional[List[int]]:
        # From: https://github.com/Haian-Jin/LVSM/blob/ebeff4989a3e1ec38fcd51ae24919d0eadf38c8f/data/dataset_scene.py#L133C1-L149C29
        if n_frames < self.input_views + self.supervise_views:
            return None

        context_indices = list(range(self.input_views))
        target_indices = list(range(self.input_views, n_frames))
        random.shuffle(target_indices)
        target_indices = target_indices[:self.supervise_views]
        indices = context_indices + target_indices
        return indices

    def __getitem__(self, _: Any) -> Dict[str, Any]:
        # Choose a random scene
        train_id = np.random.choice(self.train_ids)
        data_dir = os.path.join(self.dataset_dir, train_id)
        # data_dir = str(np.random.choice(self.data_dirs), encoding="utf-8")
        valid, meta_info = load_and_maybe_update_meta_info(data_dir, self.train_index_data[train_id])
        assert valid, f"Invalid scene: {data_dir}"

        # Select views
        n_frames = len(meta_info["frames"])
        frame_ids = self._select_views(n_frames)
        if frame_ids is None:
            return self.__getitem__(None)

        # Load frames
        loaded = load_frames_from_meta_info(
            data_dir,
            meta_info,
            frame_ids,
            patch_size=self.patch_size,
            zoom_factor=self.zoom_factor,
            random_zoom=self.random_zoom,
            random_intrinsic=True,
            random_scale=False,
        )

        # Preprocess poses
        camtoworld = torch.from_numpy(loaded["camtoworld"]).float()
        camtoworld = _normalize_poses_identity_unit_distance(
            camtoworld, ref0_idx=0, ref1_idx=self.input_views - 1
        )
        K = torch.from_numpy(loaded["K"]).float()
        image = torch.from_numpy(loaded["image"]).float()
        image_path = loaded["image_path"]

        return {
            "camtoworld": camtoworld,
            "K": K,
            "image": image,
            "image_path": image_path,
        }


class ObjaverseEvalDataset(Dataset):
    def __init__(
        self,
        test_index_file: str,
        patch_size: int = 256,
        zoom_factor: float = 1.0,  # 1.0 means disabled
        verbose: bool = False,
        first_n: Optional[int] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        input_views: int = 2,
        supervise_views: int = 3,
        render_video: bool = False,
        test_index_fp: Optional[str] = None,
        dataset_dir: str = "./data_processed/objaverse/views_release/",
    ):
        super().__init__()
        self.test_index_file = test_index_file
        self.dataset_dir = dataset_dir
        with open(test_index_file, "r") as f:
            self.test_index_data = json.load(f)
        self.test_ids = list(self.test_index_data.keys())
        self.patch_size = patch_size
        self.zoom_factor = zoom_factor
        self.input_views = input_views
        self.supervise_views = supervise_views
        self.render_video = render_video
        if self.render_video:
            raise NotImplementedError("Rendering video is not supported for Objaverse.")

        if test_index_fp is not None:
            raise NotImplementedError("Pre defined test index file is not supported for Objaverse.")

        self.test_ids = sorted(self.test_ids)

        if verbose:
            print(
                f"[EvalDataset] Found {len(self.test_ids)} scenes in the index file. "
            )

        if first_n is not None:
            self.test_ids = self.test_ids[:first_n]

        if rank is not None and world_size is not None:
            self.test_ids = self.test_ids[rank::world_size]

    def __len__(self):
        return len(self.test_ids)

    def __getitem__(self, scene_id: int) -> Dict[str, Any]:
        test_id = self.test_ids[scene_id]
        data_dir = os.path.join(self.dataset_dir, test_id)
        valid, meta_info = load_and_maybe_update_meta_info(data_dir, self.test_index_data[test_id])
        assert valid, f"Invalid scene: {data_dir}"

        # Load context and target views together
        frame_ids = np.arange(len(meta_info["frames"]))

        try:
            loaded = load_frames_from_meta_info(
                data_dir,
                meta_info,
                frame_ids,
                patch_size=self.patch_size,
                zoom_factor=self.zoom_factor,
            )
        except Exception as e:
            print(f"Error in {data_dir}: {e}. frame_ids: {frame_ids}")
            raise e

        # Preprocess poses
        camtoworld = torch.from_numpy(loaded["camtoworld"]).float()
        camtoworld = _normalize_poses_identity_unit_distance(
            camtoworld, ref0_idx=0, ref1_idx=self.input_views - 1
        )
        K = torch.from_numpy(loaded["K"]).float()
        image = torch.from_numpy(loaded["image"]).float()
        image_path = loaded["image_path"]

        return {
            "camtoworld": camtoworld,
            "K": K,
            "image": image,
            "image_path": image_path,
            "scene": scene_id,
        }


def get_raymap(K, cam2world, height=256, width=256):
    coord_transform = torch.tensor([[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    cam2world = cam2world @ coord_transform
    xs = torch.arange(width)
    ys = torch.arange(height)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    grid_x = grid_x + 0.5
    grid_y = grid_y + 0.5
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)
    
    img_xyz = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)*0.1], axis=-1) # (height * width, 3)
    cam_xyz = img_xyz @ torch.inverse(K).T # (height * width, 3)
    cam_xyz = torch.cat([cam_xyz, cam_xyz[..., :1]], axis=-1) # (height * width, 4)
    global_xyz = cam_xyz @ cam2world.T # (height * width, 4)
    global_xyz = global_xyz[..., :3] # (height * width, 3)
    global_xyz = global_xyz.reshape(height, width, 3)
    return global_xyz


if __name__ == "__main__":
    # Test the dataset.
    import glob

    """
    OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 nvs/dataset_objaverse.py
    """

    torch.distributed.init_process_group(backend="nccl")
    world_rank = int(os.environ.get("RANK", 0))

    train_index_file = "assets/objaverse_train.json"
    dataset = ObjaverseTrainDataset(
        train_index_file, verbose=True
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=4)
    data = next(iter(dataloader))
    cam2world = data["camtoworld"][0]
    image = data["image"][0]
    img_num = image.shape[0]
    for i in range(img_num):
        img = image[i]
        img = img.cpu().numpy().astype(np.uint8)
        imageio.imwrite(f"debug_vis/image_{i}.png", img)
        print(i, data["camtoworld"][0, i, :3, 3])
        # global_xyz = get_raymap(data["K"][0, i], data["camtoworld"][0, i])
        # print(global_xyz[0, 0], global_xyz[-1, 0], global_xyz[0, -1], global_xyz[-1, -1])
        # print()
    

    all_locs = data["camtoworld"][0, :, :3, 3]
    all_dist = all_locs[None, :, :] - all_locs[:, None, :]
    all_dist = all_dist.flatten(0, 1)
    print(all_dist.max(dim=0)[0])


    # print(data["image"].shape, data["K"].shape, data["camtoworld"].shape)

    # testset = EvalDataset(folder="./data_processed/realestate10k/test/", verbose=True)
    # data = testset[0]
    # print(data["image"].shape)

    torch.distributed.destroy_process_group()
