import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import cv2
import os
import re
import yaml
import math
import json
import random
import numpy as np
from PIL import Image
from typing import Callable, Union, List


class InputPadder:
    """Pads images such that dimensions are divisible by a specfical number"""

    def __init__(self, shape, divis_by=8, mode="sintel"):
        self.ht, self.wd = shape[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == "sintel":
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def pad_intri(self, intri):
        intri = intri.clone()
        intri[:, 2] = intri[:, 2] + self._pad[0]
        intri[:, 3] = intri[:, 3] + self._pad[2]
        return intri

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        return x[
            ..., self._pad[2] : ht - self._pad[3], self._pad[0] : wd - self._pad[1]
        ]

    def unpad_intri(self, intri):
        intri = intri.clone()
        intri[:, 2] = intri[:, 2] - self._pad[0]
        intri[:, 3] = intri[:, 3] - self._pad[2]
        return intri


class FlsGrid:
    def __init__(
        self,
        intri_s: torch.Tensor,
        Tc: torch.Tensor,
        Ts: torch.Tensor,
        size_ijk=[512, 512, 512],
        sigma=0.1,
    ) -> None:
        """_summary_

        Args:
            intri_s (torch.Tensor): shape [B, 3] including range, theta, phi
            Tc (torch.Tensor): pose of camera
            Ts (torch.Tensor): pose of sonar
            size_ijk (list, optional): Defaults to [512, 512, 512].
            sigma (float, optional): Defaults to 0.1.
        """
        # only consider the first data in batch to reduce calculation
        self.intri_s = intri_s[:1]  # shape [1, 3]
        self.Tc = Tc[:1]  # shape [1, 4, 4]
        self.Ts = Ts[:1]  # shape [1, 4, 4]

        self.sigma = sigma
        self.size_ijk = size_ijk
        self.device = intri_s.device

        # fls grid
        i_range = torch.arange(size_ijk[0])
        j_range = torch.arange(size_ijk[1])
        k_range = torch.arange(size_ijk[2])
        grid_i, grid_j, grid_k = torch.meshgrid(
            i_range, j_range, k_range, indexing="ij"
        )
        grids = torch.stack([grid_i, grid_j, grid_k], dim=-1)
        grids = (
            grids.to(self.device).float().unsqueeze(0)
        )  # shape [1, i_size, j_size, k_size, 3]
        grids = fls_ijk2xyz(grids, self.intri_s, self.size_ijk)
        grids = transform_coordinate(grids, self.Ts, self.Tc)
        self.grids = grids  # shape [1, i_size, j_size, k_size, 3]

    def create_grid_batch(
        self,
        depth: torch.Tensor,
        color: torch.Tensor,
        intri_c: torch.Tensor,
        intri_s: torch.Tensor,
        Tc: torch.Tensor,
        Ts: torch.Tensor,
        size_ijk=[512, 512, 512],
        sigma=0.1,
    ) -> torch.Tensor:
        B, _, H, W = depth.shape

        # fls grid
        i_range = torch.arange(size_ijk[0])
        j_range = torch.arange(size_ijk[1])
        k_range = torch.arange(size_ijk[2])
        grid_i, grid_j, grid_k = torch.meshgrid(
            i_range, j_range, k_range, indexing="ij"
        )
        grids = torch.stack([grid_i, grid_j, grid_k], dim=-1)
        grids = (
            grids.to(depth.device).float().unsqueeze(0)
        )  # shape [1, i_size, j_size, k_size, 3]
        grids = fls_ijk2xyz(grids, intri_s, size_ijk)
        grids = transform_coordinate(grids, Ts, Tc)

        # grid depth
        grid_depth = FlsGrid.depth2grid(depth, intri_s[:, 0], size_ijk[0], sigma)
        grid_depth = grid_depth.unsqueeze(dim=1)  # shape [B, 1, i_size, H, W]
        grid_color = color.unsqueeze(dim=2).expand(
            -1, -1, size_ijk[0], -1, -1
        )  # shape [B, C, i_size, H, W]
        grid_depth = torch.cat([grid_depth, grid_color], dim=1)

        # grid depth normalization
        grid_depth = depth_grid_distance_compensation(grid_depth, intri_s[:, 0])

        # create sample grids and sample
        grids = cam_xyz2uvz(grids, intri_c)  # shape [1, i_size, j_size, k_size, 3]
        grid_u = 2 * grids[..., 0] / (W - 1) - 1
        grid_v = 2 * grids[..., 1] / (H - 1) - 1
        grid_d = 2 * grids[..., 2] / align_left(intri_s[:, 0], grids[..., 2]) - 1
        grids = torch.stack([grid_u, grid_v, grid_d], dim=-1)
        grid_fls = F.grid_sample(
            grid_depth, grids, align_corners=True
        )  # shape [B, 1+C, i_size, j_size, k_size]
        return grid_fls

    def create_grid(
        self, depth: torch.Tensor, color: torch.Tensor, intri_c: torch.Tensor
    ) -> torch.Tensor:
        """create fls grid in fls coordinate from depth map

        Args:
            depth (torch.Tensor): shape [B, 1, H, W]
            color (torch.Tensor): shape [B, C, H, W]
            intri_c (torch.Tensor): shape [B, 4] including fx, fy, cx, cy

        Returns:
            torch.Tensor: _description_
        """
        B, _, H, W = depth.shape

        # grid depth
        grid_depth = FlsGrid.depth2grid(
            depth, self.intri_s[:, 0], self.size_ijk[0], self.sigma
        )
        grid_depth = grid_depth.unsqueeze(dim=1)  # shape [B, 1, i_size, H, W]
        grid_color = color.unsqueeze(dim=2).expand(
            -1, -1, self.size_ijk[0], -1, -1
        )  # shape [B, C, 1, H, W]
        grid_depth = torch.cat([grid_depth, grid_color], dim=1)

        # grid depth normalization
        grid_depth = depth_grid_distance_compensation(grid_depth, self.intri_s[0, 0])

        # create sample grids and sample
        grids = cam_xyz2uvz(self.grids, intri_c)  # shape [1, i_size, j_size, k_size, 3]
        grid_u = 2 * grids[..., 0] / (W - 1) - 1
        grid_v = 2 * grids[..., 1] / (H - 1) - 1
        grid_d = 2 * grids[..., 2] / self.intri_s[:, 0] - 1
        grids = torch.stack([grid_u, grid_v, grid_d], dim=-1)
        grid_fls = F.grid_sample(
            grid_depth, grids, align_corners=True
        )  # shape [B, 1+C, i_size, j_size, k_size]
        return grid_fls

    @staticmethod
    def depth2grid(
        depth: torch.Tensor, max_depth: torch.Tensor, grid_size=512, sigma=0.1
    ) -> torch.Tensor:
        """convert depth to depth distribution grid

        Args:
            depth (torch.Tensor): shape [B, 1, H, W]
            max_depth (int, optional): shape [B, 1]
            grid_size (int, optional): Defaults to 512.
            sigma (float, optional): Defaults to 0.1.

        Returns:
            torch.Tensor: shape [B, grid_size, H, W]
        """
        max_depth = align_left(max_depth, depth)
        voxel_size = max_depth / (grid_size - 1)
        grid = (
            torch.arange(grid_size).reshape(1, grid_size, 1, 1).to(depth.device).float()
        )
        grid = (grid * voxel_size - depth).square()  # square distance
        grid = torch.exp(-grid / (2 * sigma**2))  # gassue distribution
        grid = grid / (1e-6 + grid.sum(dim=1, keepdim=True))  # normalization

        return grid  # shape [B, grid_size, H, W]


def imgs_intensity_align(
    ref: torch.Tensor,
    src: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """align alone r, imgs shape (B, 1, R, Theta)

    Args:
        ref (torch.Tensor): shape (B, 1, H, W)
        src (torch.Tensor): shape (B, 1, H, W)
        mask (torch.Tensor): shape (B, 1, H, W)

    Returns:
        List[torch.Tensor]: shape (B, 1, H, W)
    """
    # aligen intensity
    ref_sum_theta = (ref * mask).sum(dim=-1, keepdim=True)
    src_sum_theta = (src * mask).sum(dim=-1, keepdim=True)
    scale_r = ref_sum_theta / (src_sum_theta + 1e-6)  # shape (B, 1, R, 1)
    src = src * scale_r.detach()
    return src


def depth_grid_distance_compensation(
    grid: torch.Tensor, max_distance: torch.Tensor
) -> torch.Tensor:
    """Normalize the occupancy grid generated from depth image

    Args:
        grid (torch.Tensor): shape (B, ..., D, H, W)
        max_distance (float): torch.Tensor shape (B)

    Returns:
        torch.Tensor: _description_
    """
    D, H, W = grid.shape[-3:]
    max_distance = align_left(max_distance, grid)  # shape (B, ..., 1, 1, 1)
    distance_inter = max_distance / (D - 1)  # shape (B, ..., 1, 1, 1)

    distances = (
        torch.linspace(0, D - 1, D).reshape(-1, 1, 1).to(grid.device)
    )  # shape (D, 1, 1)
    distances = distance_inter * distances  # shape (B, ..., D, 1, 1)

    compensation_factors = distances  # ** 2
    compensation_factors /= torch.max(compensation_factors)
    return grid * compensation_factors


def fls_distance_normalization(
    imgs: torch.Tensor,
) -> torch.Tensor:
    """Apply gain normalization alone range for a fls image

    Args:
        imgs (torch.Tensor): shape (..., r_size, theta_size)

    Returns:
        torch.Tensor: normalized fls image
    """
    line_sum = [imgs[..., l, :].sum() for l in range(imgs.shape[-2])]
    line_tar = sum(line_sum) / len(line_sum)
    for l in range(imgs.shape[-2]):
        rate = line_tar / line_sum[l]
        imgs[..., l, :] = imgs[..., l, :] * rate
    return imgs


def fls_distance_compensation(imgs: torch.Tensor, max_distance: float) -> torch.Tensor:
    """Apply inverse square law distance compensation to a fls image

    Args:
        imgs (torch.Tensor): shape (..., r_size, theta_size)
        max_distance (float):

    Returns:
        torch.Tensor: normalized fls image
    """
    r_size, theta_size = imgs.shape[-2:]

    distances = torch.linspace(0, max_distance, r_size).reshape(-1, 1)  # [r_size, 1]
    distances = distances.to(imgs.device)
    compensation_factors = distances**2
    compensation_factors /= torch.max(compensation_factors)
    return imgs * compensation_factors


def align_left(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Align the dimensions of tensor src to the dimensions of tensor tgt from the left.

    Args:
        src (torch.Tensor): The tensor to be aligned.
        tgt (torch.Tensor): The tensor to align to.

    Returns:
        torch.Tensor: Aligned tensor src with the same number of dimensions as tgt.
    """
    shape_src = list(src.shape)
    shape_tgt = list(tgt.shape)
    if len(shape_src) == len(shape_tgt):
        return src

    num_dims_to_add = len(shape_tgt) - len(shape_src)
    new_shape = shape_src + [1] * num_dims_to_add
    return src.reshape(new_shape)


def cam_xyz2uvz(
    points: torch.Tensor,
    intri_c: torch.Tensor,
) -> torch.Tensor:
    """convert [x, y, z] to camera [u, v, d]

    Args:
        points (torch.Tensor): shape [B, ..., 3]
        intri_c (torch.Tensor): shape [B, 4] with fx, fy, cx, cy

    Returns:
        torch.Tensor: shape [B, ..., 3]
    """
    fx = align_left(intri_c[:, 0], points[..., 0])
    fy = align_left(intri_c[:, 1], points[..., 0])
    cx = align_left(intri_c[:, 2], points[..., 0])
    cy = align_left(intri_c[:, 3], points[..., 0])
    u = (fx * points[..., 0] + cx * points[..., 2]) / points[..., 2]
    v = (fy * points[..., 1] + cy * points[..., 2]) / points[..., 2]
    z = points[..., 2].expand_as(v)
    return torch.stack([u, v, z], dim=-1)


def cam_uvz2xyz(
    points: torch.Tensor,
    intri_c: torch.Tensor,
) -> torch.Tensor:
    """convert camera [u, v, d] to [x, y, z]

    Args:
        points (torch.Tensor): shape [B, ..., 3]
        intri_c (torch.Tensor): shape [B, 4] with fx, fy, cx, cy

    Returns:
        torch.Tensor: shape [B, ..., 3]
    """
    fx = align_left(intri_c[:, 0], points[..., 0])
    fy = align_left(intri_c[:, 1], points[..., 0])
    cx = align_left(intri_c[:, 2], points[..., 0])
    cy = align_left(intri_c[:, 3], points[..., 0])
    x = (points[..., 0] - cx) / fx * points[..., 2]
    y = (points[..., 1] - cy) / fy * points[..., 2]
    z = points[..., 2]
    return torch.stack([x, y, z], dim=-1)


def fls_xyz2ijk(
    points: torch.Tensor,
    intri_s: torch.Tensor,  # range of r, theta, phi
    size_ijk=[512, 512, 512],
) -> torch.Tensor:
    """convert x, y, z to i, j, k of fls

    Args:
        points (torch.Tensor): shape [B, ..., 3]
        intri_s (torch.Tensor): shape [B, 3]
        size_ijk (List): Defaults to 512.
        phisize_ijk (list, optional): Defaults to [512, 512, 512].

    Returns:
        torch.Tensor: shape [B, ..., 3]
    """
    r_range = align_left(intri_s[:, 0], points[..., 0])
    t_range = align_left(intri_s[:, 1], points[..., 0])
    p_range = align_left(intri_s[:, 2], points[..., 0])

    # to r, theta, phi (radian)
    r = torch.sqrt(points[..., 0] ** 2 + points[..., 1] ** 2 + points[..., 2] ** 2)
    t = torch.arctan2(points[..., 1], points[..., 0])
    p = torch.arctan2(
        -points[..., 2], (points[..., 0] ** 2 + points[..., 1] ** 2).sqrt()
    )

    i = r / r_range * (size_ijk[0] - 1)
    j = (t.rad2deg() / t_range + 0.5) * (size_ijk[1] - 1)
    k = (p.rad2deg() / p_range + 0.5) * (size_ijk[2] - 1)
    return torch.stack([i, j, k], dim=-1)


def fls_ijk2xyz(
    points: torch.Tensor,
    intri_s: torch.Tensor,  # range of r, theta, phi
    size_ijk=[512, 512, 512],
) -> torch.Tensor:
    """convert i, j, k of fls to x, y, z

    Args:
        points (torch.Tensor): shape [B, ..., 3]
        intri_s (torch.Tensor): shape [B, 3]
        size (int, optional): Defaults to 512.

    Returns:
        torch.Tensor: shape [B, ..., 3]
    """
    r_range = align_left(intri_s[:, 0], points[..., 0])
    t_range = align_left(intri_s[:, 1], points[..., 0])
    p_range = align_left(intri_s[:, 2], points[..., 0])

    # to r, theta, phi (degree)
    r = (
        points[..., 0] / (size_ijk[0] - 1) * r_range
    )  # range [0, grid_size-1] -> [0, r_rage]
    t = (
        points[..., 1] / (size_ijk[1] - 1) - 0.5
    ) * t_range  # range [0, grid_size-1] -> [-theta_range/2, theta_range/2]
    p = (
        points[..., 2] / (size_ijk[2] - 1) - 0.5
    ) * p_range  # range [0, grid_size-1] -> [-phi_range/2, phi_range/2]

    x = r * torch.cos(t.deg2rad()) * torch.cos(p.deg2rad())
    y = r * torch.sin(t.deg2rad()) * torch.cos(p.deg2rad())
    z = -r * torch.sin(p.deg2rad())
    return torch.stack([x, y, z], dim=-1)


def transform_coordinate(
    points: torch.Tensor,
    T_current: torch.Tensor,
    T_target: torch.Tensor,
) -> torch.Tensor:
    """transform points from current to target coordinate

    Args:
        points (torch.Tensor): shape [B, ..., 3]
        T_currect (torch.Tensor): [B, 4, 4]
        T_target (torch.Tensor): [B, 4, 4]

    Returns:
        torch.Tensor: shape [B, ..., 3]
    """
    org_shape = points.shape
    points = points.reshape([org_shape[0], -1, 3])  # [B, N, 3]

    # to homogeneous
    B, N, _ = points.shape
    ones = torch.ones([B, N, 1]).to(points.device)
    points = torch.cat([points, ones], dim=-1)  # [B, N, 4]

    # tranform to base then to camera
    T = torch.matmul(T_target.inverse(), T_current)  # shape [B, 4, 4]
    points = torch.bmm(points, T.transpose(1, 2))[..., :-1]  # [B, N, 3]
    points = points.reshape(org_shape)
    return points


def get_grad(img: torch.Tensor) -> torch.Tensor:
    """calculate gradient image with sobel"""
    C = img.shape[1]
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
    sobel_x = sobel_x[None, None, ...].repeat(C, 1, 1, 1).to(img.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
    sobel_y = sobel_y[None, None, ...].repeat(C, 1, 1, 1).to(img.device)
    grad_x = F.conv2d(img, sobel_x, padding=1, groups=C)
    grad_y = F.conv2d(img, sobel_y, padding=1, groups=C)
    grad = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    return grad


def get_grad2(img: torch.Tensor) -> torch.Tensor:
    """calculate second-order with laplacian"""
    C = img.shape[1]
    laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
    laplacian = laplacian[None, None, ...].repeat(C, 1, 1, 1).to(img.device)
    grad = F.conv2d(img, laplacian, padding=1, groups=C)
    grad = torch.sqrt(grad**2 + 1e-6)  # Change the type from float16 to float32
    return grad


def white_balance_enchancement(img: torch.Tensor):
    """Apply white balance to a batch of images."""
    channel_means = img.mean(dim=(2, 3), keepdim=True)  # Shape: (B, C, 1, 1)
    mean_intensity = channel_means.mean(dim=1, keepdim=True)  # Shape: (B, 1, 1, 1)
    scale_factors = mean_intensity / channel_means  # Shape: (B, C, 1, 1)
    balanced_image = img * scale_factors
    balanced_image = balanced_image.clamp(0.0, 1.0)
    return balanced_image


def max_enchancement(img: torch.Tensor):
    """Normalization by maximum"""
    B, C = img.shape[:2]
    max_vals = img.view(B, C, -1).max(dim=2)[0].reshape(B, C, 1, 1)
    return img / max_vals


def collaborative_enhancement(
    imgl: torch.Tensor, imgr: torch.Tensor, types=["white", "max"]
):
    """collaborative enhancement with type "max", "white" mode"""
    imglr = torch.cat([imgl, imgr], dim=-1)
    for type in types:
        if type == "white":
            imglr = white_balance_enchancement(imglr)
        elif type == "max":
            imglr = max_enchancement(imglr)

    W = imglr.shape[-1]
    return imglr[..., : W // 2], imglr[..., W // 2 :]


def create_mask(grid: torch.Tensor, thread=2e-1) -> torch.Tensor:
    """create mask from fls grid, shape [B, phi_size, r_size, theta_size]"""
    mask = grid.max(dim=-3, keepdim=True)[0]
    mask[mask >= thread] = 1
    mask[mask <= thread] = 0
    return mask.detach()


def warping_stereo(
    img: torch.Tensor,
    disp: torch.Tensor,
    target="left",
) -> torch.Tensor:
    """warp a image from given disp"""
    B, _, H, W = img.shape
    grid_i, grid_j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    grid_i = grid_i.repeat([B, 1, 1]).to(img.device).float()  # (b, h, w)
    grid_j = grid_j.repeat([B, 1, 1]).to(img.device).float()  # (b, h, w)

    if target == "left":
        grid_j = grid_j - disp[:, 0]  # (b, h, w) - (b, 1, h, w)[:,0]
    elif target == "right":
        grid_j = grid_j + disp[:, 0]  # (b, h, w) - (b, 1, h, w)[:,0]

    grid_i = 2 * grid_i / (H - 1) - 1
    grid_j = 2 * grid_j / (W - 1) - 1
    grid = torch.stack([grid_j, grid_i], dim=-1)
    return F.grid_sample(img, grid, align_corners=True)


def disp2depth(
    disp: torch.Tensor,
    fx: torch.Tensor,
    baseline: torch.Tensor,
    max_depth=80,
) -> torch.Tensor:
    """Convert disparity to depth

    Args:
        disp (torch.Tensor): shape [B, 1, H, W] or [1, H, W]
        fx (torch.Tensor): shape [B] or [1]
        baseline (torch.Tensor): shape [B] or [1]
        max (int, optional): Defaults to 80.

    Returns:
        torch.Tensor: shape [B, 1, H, W]
    """
    fx = align_left(fx, disp)
    baseline = align_left(baseline, disp)

    mask = disp >= 0
    depth = torch.zeros_like(disp)
    depth[mask] = (fx * baseline / (disp + 1e-8))[mask]
    depth[depth > max_depth] = max_depth
    return depth


def SSIM(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """similiary"""
    C1 = 0.01**2
    C2 = 0.03**2
    mu_x = F.avg_pool2d(x, 3, 1, padding=1)
    mu_y = F.avg_pool2d(y, 3, 1, padding=1)
    sigma_x = F.avg_pool2d(x**2, 3, 1, padding=1) - mu_x**2
    sigma_y = F.avg_pool2d(y**2, 3, 1, padding=1) - mu_y**2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, padding=1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    ssim = ssim_n / ssim_d

    return ssim


def sonar2color(sonar_image: torch.Tensor, theta_range: torch.Tensor) -> torch.Tensor:
    """convert sonar array  to color images

    Args:
        sonar_image (torch.Tensor): shape [1, r_size, theta_size]
        theta_range (torch.Tensor): shape [1]

    Returns:
        torch.Tensor: shape [1, H, W]
    """
    r_size, theta_size = sonar_image.shape[-2:]
    H = r_size
    W = int(2 * r_size * math.sin(theta_range / 2))
    origin_i, origin_j = H - 1, (W - 1) / 2
    # create meshgrid
    grid_i, grid_j = torch.meshgrid(
        torch.arange(H), torch.arange(W), indexing="ij"
    )  # [H, W]
    grid_i = grid_i.to(sonar_image.device)  # [H, W]
    grid_j = grid_j.to(sonar_image.device)  # [H, W]
    # convert to polar index
    grid_r = ((grid_i - origin_i) ** 2 + (grid_j - origin_j) ** 2).sqrt()  # [H, W]
    grid_t = torch.arctan2(origin_j - grid_j, origin_i - grid_i)  # [H, W]
    grid_t = (grid_t / torch.pi * 180 / theta_range + 0.5) * theta_size  # [H, W]
    # convert to [-1, 1] and sample
    grid_r = 2 * grid_r / (r_size - 1) - 1
    grid_t = 2 * grid_t / (theta_size - 1) - 1
    grid = torch.stack((grid_t, grid_r), dim=-1)  # [H, W, 2]
    sonar_image_view = F.grid_sample(
        sonar_image[None], grid[None], align_corners=True
    )  # [1, 1, H, W]
    return sonar_image_view[0]  # [1, H, W]


def tensorboard_write_dict(
    writer: SummaryWriter, metrics: dict, step: int, mode="scalar"
):
    """write dict on tensorboard"""
    assert mode == "scalar" or mode == "figure" or mode == "image"
    if mode == "scalar":
        for k in metrics.keys():
            writer.add_scalar(k, metrics[k], step)
    elif mode == "figure":
        for k in metrics.keys():
            writer.add_figure(k, metrics[k], step)
    elif mode == "image":
        for k in metrics.keys():
            writer.add_image(k, metrics[k], step)


def worker_init_fn(worker_id):
    """fix all the seeds for each thread"""
    seed = torch.initial_seed() % 2**32
    torch.manual_seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def fix_seeds(seed=1234):
    """fix all the seeds in pytorch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    # torch.use_deterministic_algorithms(True)


def save_random_states():
    """save random state"""
    state_dict = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state_dict["torch_cuda_random"] = torch.cuda.get_rng_state_all()
    return state_dict


def restore_random_states(state_dict):
    """restore random state"""
    random.setstate(state_dict["python_random"])
    np.random.set_state(state_dict["numpy_random"])
    torch.set_rng_state(state_dict["torch_random"])
    if "torch_cuda_random" in state_dict and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state_dict["torch_cuda_random"])


def read_PFM(file: str) -> np.ndarray:
    """read pfm file"""
    file = open(file, "rb")

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b"PF":
        color = True
    elif header == b"Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(rb"^(\d+)\s(\d+)\s$", file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def read_generally(file_path: str) -> Union[np.ndarray, Image.Image]:
    """general read function for image and disparty"""
    ext = os.path.splitext(file_path)[-1]
    if ext == ".png" or ext == ".jpeg" or ext == ".ppm" or ext == ".jpg":
        return Image.open(file_path)
    elif ext == ".bin" or ext == ".raw" or ext == ".npy":
        return np.load(file_path)
    elif ext == ".pfm":
        disp = read_PFM(file_path).astype(np.float32)
        if len(disp.shape) == 2:
            return disp
        else:
            return disp[:, :, :-1]
    return []


def read_camera_yaml(file_path: str):
    """read camera intrinsics and extrinsics from yaml file"""
    with open(file_path, "r") as file:
        param = yaml.safe_load(file)
        extri = torch.tensor(param["T^B_C"]["data"]).reshape(4, 4)
        intri = torch.tensor(param["intrinsics"])
    return extri, intri


def read_stereo_yaml(file_path_l: str, file_path_r: str):
    """read stereo camera intrinsics and extrinsics"""
    extri_l, intri_l = read_camera_yaml(file_path_l)
    extri_r, intri_r = read_camera_yaml(file_path_r)
    baseline = torch.norm(extri_r[:3, 3] - extri_l[:3, 3])
    assert (intri_l == intri_r).all(), "stereo camera is not undistorted"
    return extri_l, extri_r, intri_l, baseline


def read_fls_yaml(file_path: str):
    """read fls sonar intrinsics and extrinsics from yaml file"""
    with open(file_path, "r") as file:
        param = yaml.safe_load(file)
        extri = torch.tensor(param["T^B_S"]["data"]).reshape(4, 4)
        intri = torch.tensor(
            [
                param["max_distance"],
                param["horizontal_degree"],
                param["vertical_degree"],
            ]
        )
    return extri, intri


@torch.no_grad()
def gpu_warmup(fun: Callable, *args):
    """GPU warmup function for time tests"""
    # create event object
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # GPU warm-up for 3 seconds
    start.record()
    while True:
        _ = fun(*args)
        end.record()
        torch.cuda.synchronize()
        if start.elapsed_time(end) > 3000:
            break
    print("GPU warm-up end. ")


@torch.no_grad()
def gpu_speed_test(fun: Callable, loop: int, name: str, *args):
    """test inference time on gpu

    Args:
        fun (Callable): test object
        loop (int): loop times for average
        name (str): name for print
    """
    # create event object
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # GPU warm-up
    gpu_warmup(fun, *args)
    # Testing time through a loop
    timings = np.zeros([loop, 1])
    with torch.no_grad():
        for i in range(loop):
            start.record()
            fun(*args)
            end.record()
            # Waiting for GPU synchronization to complete
            torch.cuda.synchronize()
            runtime = start.elapsed_time(end)
            timings[i] = runtime
    mean_syn = np.sum(timings) / loop
    std_syn = np.std(timings)
    print(
        name, "mean %.4f ms, std %.4f ms, under %d loops." % (mean_syn, std_syn, loop)
    )
