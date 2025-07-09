import torch
import torchvision.transforms.v2.functional as v2F

import time
import random
import logging
import numpy as np
from tqdm import tqdm

from options import Options
from show import create_stereo
from src.datasets import DaveSonar
from src.utils import *


def random_multipath(imgs: torch.Tensor, shift=0):
    B, C, H, W = imgs.shape
    dx = random.randint(-shift, shift)
    dy = random.randint(-shift, shift)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    grid_x = grid_x.to(imgs.device)
    grid_y = grid_y.to(imgs.device)
    shifted_grid_x = grid_x + (dx / W) * 2
    shifted_grid_y = grid_y + (dy / H) * 2

    sample_grid = torch.stack([shifted_grid_x, shifted_grid_y], dim=-1).repeat(
        B, 1, 1, 1
    )  # [B, H, W, 2]

    shifted_imgs = torch.nn.functional.grid_sample(
        imgs, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return (imgs + 0.7 * shifted_imgs) / (1 + 0.7)


def random_occlusion(img: torch.Tensor, occlusion=0.1):
    H, W = img.shape[-2:]
    dx_rate = torch.rand([]) * (1 - occlusion) + occlusion
    dy_rate = occlusion / dx_rate
    dx = int(W * dx_rate)
    dy = int(H * dy_rate)
    x0 = (torch.rand([]) * (W - dx)).int()
    y0 = (torch.rand([]) * (H - dy)).int()
    img[:, :, y0 : y0 + dy, x0 : x0 + dx] = 0
    return img


def light_attenuation(img: torch.Tensor, depth: torch.Tensor, turbidity=1.0):
    beta_r, beta_g, beta_b = 0.20 * turbidity, 0.14 * turbidity, 0.10 * turbidity
    attenuation_r = torch.exp(-beta_r * depth)
    attenuation_g = torch.exp(-beta_g * depth)
    attenuation_b = torch.exp(-beta_b * depth)
    attenuation = torch.cat([attenuation_r, attenuation_g, attenuation_b], dim=1)
    return torch.clamp(img * attenuation, 0, 1)


def disp_metric(disp_pr: torch.Tensor, disp_gt: torch.Tensor, valid_mask: torch.Tensor):
    # epe
    epe = (disp_pr - disp_gt).abs().flatten()
    epe = epe[valid_mask].mean().item()
    # d1
    d1 = ((disp_pr - disp_gt).abs() > 1.0).flatten()
    d1 = d1[valid_mask].cpu().numpy()
    return epe, d1


def depth_metric(
    depth_pr: torch.Tensor, depth_gt: torch.Tensor, valid_mask: torch.Tensor
):
    diff_abs = (depth_pr - depth_gt).abs()
    diff_log = (depth_pr + 1e-6).log() - (depth_gt + 1e-6).log()
    # rel
    abs_rel = (diff_abs / depth_gt).flatten()
    abs_rel = abs_rel[valid_mask].mean().item()
    # sq rel
    sq_rel = (diff_abs / depth_gt).square().flatten()
    sq_rel = sq_rel[valid_mask].mean().item()
    # rmse
    rmse = diff_abs.square().flatten()
    rmse = rmse[valid_mask].mean().sqrt().item()
    # rmse log
    rmse_log = diff_log.square().flatten()
    rmse_log = rmse_log[valid_mask].mean().sqrt().item()
    # delta
    delta = torch.max(depth_pr / depth_gt, depth_gt / depth_pr)
    delta1 = (delta < 1.25).float().flatten()
    delta1 = delta1[valid_mask].mean().item()
    delta2 = (delta < 1.25**2).float().flatten()
    delta2 = delta2[valid_mask].mean().item()
    delta3 = (delta < 1.25**3).float().flatten()
    delta3 = delta3[valid_mask].mean().item()
    return abs_rel, sq_rel, rmse, rmse_log, delta1, delta2, delta3


@torch.no_grad()
def valid_dave(
    model,
    border=10,
    iters=3,
    shift=0,
    noise=0,
    contrast=1,
    sigma=0,
    occlusion=0,
    distortion=0,
    displacement=0,
    fls_movement=0,
):
    """Peform validation using the FlyingThings3D (TEST) split"""
    model.eval()
    device = str(next(model.parameters()).device)
    val_dataset = DaveSonar(scenes=["4"], train=False)

    random_state = save_random_states()
    fix_seeds(1234)

    d1_list, epe_list = [], []
    abs_rel_list, sq_rel_list = [], []
    rmse_list, rmse_log_list = [], []
    delta1_list, delta2_list, delta3_list = [], [], []
    for val_id in tqdm(range(len(val_dataset))):
        data = val_dataset[val_id]
        imgl = data["imgl"][None].to(device)
        imgr = data["imgr"][None].to(device)
        imgs = data["imgs"][None].to(device)
        intri_c = data["intri_c"][None].to(device)
        intri_s = data["intri_s"][None].to(device)
        baseline = data["baseline"][None].to(device)
        Tc = data["Tc"][None].to(device)
        Ts = data["Ts"][None].to(device)
        disp_gt = data["disp"][None].to(device)
        valid_gt = data["valid"][None].to(device)

        device_type = "cuda" if "cuda" in device else "cpu"
        with torch.autocast(enabled="cuda" in device, device_type=device_type):
            depth_gt = disp2depth(disp_gt, intri_c[:, 0], baseline)
            # multi path on imgs
            imgs = random_multipath(imgs, shift)
            # noise on imgs
            imgs = imgs + noise * torch.randn_like(imgs)
            imgs = imgs.clamp(0, 1)
            # contrast
            imgl = v2F.adjust_contrast(imgl, contrast)
            imgr = v2F.adjust_contrast(imgr, contrast)
            # blur
            if sigma > 0:
                imgl = v2F.gaussian_blur(imgl, 15, sigma)
                imgr = v2F.gaussian_blur(imgr, 15, sigma)
            # occu
            if occlusion > 0:
                imgr = random_occlusion(imgr, occlusion)
            # fx change
            intri_distortion = intri_c.clone()
            if distortion != 0:
                intri_distortion[:, 0] = intri_distortion[:, 0] * (1 + distortion)
            # baseline change
            baseline_displacement = baseline.clone()
            if displacement != 0:
                baseline_displacement = baseline_displacement * (1 + displacement)
            # fls move in left camera coordinate
            fls_move = torch.tensor(fls_movement).to(device)
            if (fls_move != 0).any():
                Ts_offset = (Tc.inverse() @ Ts).float()
                Ts_offset[..., :3, -1] += fls_move
                Ts_move = (Tc @ Ts_offset).float()
            else:
                Ts_move = Ts.clone()

            # prediction
            padder = InputPadder(imgl.shape, divis_by=32)
            imgl_pad, imgr_pad = padder.pad(imgl, imgr)
            intri_pad = padder.pad_intri(intri_distortion)
            disp_pr = model(
                imgl_pad,
                imgr_pad,
                imgs,
                baseline_displacement,
                intri_pad,
                intri_s,
                Tc,
                Ts_move,
                iters=iters,
                train=False,
            )
            disp_pr = padder.unpad(disp_pr)
            depth_pr = disp2depth(
                disp_pr, intri_distortion[:, 0], baseline_displacement
            )

        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
        valid_gt[..., :border] = 0
        valid_gt[..., -border:] = 0
        valid_mask = valid_gt.flatten() >= 0.5
        # disp
        epe, d1 = disp_metric(disp_pr, disp_gt, valid_mask)
        epe_list.append(epe)
        d1_list.append(d1)
        # depth
        abs_rel, sq_rel, rmse, rmse_log, delta1, delta2, delta3 = depth_metric(
            depth_pr, depth_gt, valid_mask
        )
        abs_rel_list.append(abs_rel)
        sq_rel_list.append(sq_rel)
        rmse_list.append(rmse)
        rmse_log_list.append(rmse_log)
        delta1_list.append(delta1)
        delta2_list.append(delta2)
        delta3_list.append(delta3)

    epe = np.array(epe_list).mean()
    d1 = np.concatenate(d1_list).mean() * 100
    abs_rel = np.mean(abs_rel_list)
    sq_rel = np.mean(sq_rel_list)
    rmse = np.mean(rmse_list)
    rmse_log = np.mean(rmse_log_list)
    delta1 = np.mean(delta1_list)
    delta2 = np.mean(delta2_list)
    delta3 = np.mean(delta3_list)

    restore_random_states(random_state)
    print(f"Validation DAVE Sonar: EPE: {epe:.3f}, D1: {d1:.3f}")
    print(
        f"Validation DAVE Sonar: AbsRel: {abs_rel:.3f}, SqRel: {sq_rel:.3f}, RMSE: {rmse:.3f}, RMSE log: {rmse_log:.3f}, \delta^1: {delta1:.3f}, \delta^2: {delta2:.3f}, \delta^3: {delta3:.3f}"
    )
    result_dict = {
        "dave/epe": epe,
        "dave/d1": d1,
        "dave/abs-rel": abs_rel,
        "dave/sq-rel": sq_rel,
        "dave/rmse": rmse,
        "dave/rmse-log": rmse_log,
        "dave/delta1": delta1,
        "dave/delta2": delta2,
        "dave/delta3": delta3,
    }
    return result_dict


if __name__ == "__main__":
    # settings
    args = Options().parse()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    # model
    model = create_stereo(args, train=False)

    for valid_set in args.valid_sets:

        if valid_set == "dave_sonar":
            valid_dave(
                model,
                args.border,
                iters=args.update_iters,
                shift=args.valid_multipath_shift,
                noise=args.valid_noise_level,
                contrast=args.valid_contrast_coff,
                sigma=args.valid_blur_sigma,
                occlusion=args.valid_occlusion,
                distortion=args.valid_fx_distortion,
                displacement=args.valid_displacement,
                fls_movement=args.valid_fls_movement,
            )
