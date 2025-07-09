import torch
from typing import List, Dict

from src.utils import *


def unary_loss(
    img: torch.Tensor,
    rec: torch.Tensor,
    mask=None,
    writer=None,
    log_name="loss_vis/unary",
    weight=[0.8, 0.1, 0.1],
) -> torch.Tensor:
    """unary loss"""
    if mask is None:
        mask = torch.ones_like(img).detach()

    img_grad = get_grad(img)
    rec_grad = get_grad(rec)

    ssim_loss = (1 - SSIM(img, rec)).mean(dim=1, keepdim=True) / 2 * mask
    rec_loss = (img - rec).abs().mean(dim=1, keepdim=True) * mask
    grad_loss = (img_grad - rec_grad).abs().mean(dim=1, keepdim=True) * mask

    if writer is not None:
        from show import show_plt

        fig = show_plt(
            img[0],
            rec[0],
            mask[0],
            ssim_loss[0],
            rec_loss[0],
            grad_loss[0] / grad_loss[0].max(),
            show=False,
        )
        writer({log_name: fig}, mode="figure")
    return (
        weight[0] * ssim_loss.sum()
        + weight[1] * rec_loss.sum()
        + weight[2] * grad_loss.sum()
    ) / mask.sum()


def loop_consistency_loss(
    img: torch.Tensor,
    rerec: torch.Tensor,
    mask=None,
    writer=None,
    log_name="loss_vis/loop",
) -> torch.Tensor:
    """loop consistency loss"""
    if mask is None:
        mask = torch.ones_like(img).detach()

    # loss
    loss = (img - rerec).abs().mean(dim=1, keepdim=True) * mask

    if writer is not None:
        from show import show_plt

        fig = show_plt(img[0], rerec[0], loss[0], mask[0], show=False)
        writer({log_name: fig}, mode="figure")
    return loss.sum() / mask.sum()


def regularization_loss(
    img: torch.Tensor,
    disp: torch.Tensor,
    mask=None,
    writer=None,
    log_name="loss_vis/reg",
) -> torch.Tensor:
    """regularization loss"""
    if mask is None:
        mask = torch.ones_like(img).detach()

    img_grad2 = get_grad2(img)
    disp_grad2 = get_grad2(disp)
    weight = torch.exp(-1 * img_grad2.mean(dim=1, keepdim=True))
    reg_loss = (disp_grad2 * weight).mean(dim=1, keepdim=True) * mask
    reg_loss = reg_loss.clamp(0, 1)

    if writer is not None:
        from show import show_plt

        fig = show_plt(img_grad2[0], weight[0], disp_grad2[0], reg_loss[0], show=False)
        writer({log_name: fig}, mode="figure")
    return reg_loss.sum() / mask.sum()


def unsupervised_stereo_loss(
    imgl: torch.Tensor,
    imgr: torch.Tensor,
    displ: torch.Tensor,
    dispr: torch.Tensor,
    weights: List[float],
    border=10,
    writer=None,
) -> torch.Tensor:
    # mask
    mask = torch.ones_like(displ).detach()
    if border > 0:
        mask[:, :, :, :border] = 0
        mask[:, :, :, -border:] = 0

    # reconstruction from stereo
    recl = warping_stereo(imgr, displ, target="left")
    recr = warping_stereo(imgl, dispr, target="right")
    rerecl = warping_stereo(recr, displ, target="left")

    unary_term = unary_loss(imgl, recl, mask, writer, "loss_vis/stereo_unary")
    loop_term = loop_consistency_loss(
        imgl, rerecl, mask, writer, "loss_vis/stereo_loop"
    )
    reg_term = regularization_loss(imgl, displ, mask, writer, "loss_vis/stereo_reg")
    loss = weights[0] * unary_term + weights[1] * loop_term + weights[2] * reg_term

    if writer is not None:
        from show import show_plt

        W = imgl.shape[-1]
        fig = show_plt(
            imgl[0],
            displ[0] / (W - 1),
            recl[0],
            imgr[0],
            dispr[0] / (W - 1),
            recr[0],
            show=False,
        )
        writer({"loss_vis/stereo": fig}, mode="figure")
    return loss


def unsupervised_sonar_loss(
    imgs: torch.Tensor,
    imgs_pr: torch.Tensor,
    weights: List[float],
    mask=None,
    writer=None,
) -> torch.Tensor:
    imgs = F.interpolate(imgs, imgs_pr.shape[-2:], mode="bilinear", align_corners=True)

    # mask
    if mask is None:
        mask = torch.ones_like(imgs).detach()
        mask[imgs_pr <= 1e-3] = 0

    # align alone r, imgs shape (B, 1, R, Theta)
    imgs_pr = imgs_intensity_align(imgs, imgs_pr, mask)

    unary_term = unary_loss(
        imgs,
        imgs_pr,
        mask,
        writer,
        "loss_vis/sonar_unary",
    )
    loss = weights[3] * unary_term
    return loss


def unsupervised_loss(
    batch: Dict[str, torch.Tensor],
    output: Dict[str, torch.Tensor],
    args: Dict,
    writer=None,
) -> torch.Tensor:
    loss = 0
    n_disp_pr = len(output["displ_list"])

    if writer is not None:
        from show import show_plt

        W = batch["imgl"].shape[-1]
        fig = show_plt(
            output["displ_list"][0][0] / (W - 1),
            output["displ_list"][1][0] / (W - 1),
            output["displ_list"][2][0] / (W - 1),
            output["displ_list"][3][0] / (W - 1),
            output["dispr_list"][0][0] / (W - 1),
            output["dispr_list"][1][0] / (W - 1),
            output["dispr_list"][2][0] / (W - 1),
            output["dispr_list"][3][0] / (W - 1),
            show=False,
        )
        writer({"loss_vis/disp": fig}, mode="figure")

    # imgl, imgr enchance
    imgl_enc, imgr_enc = collaborative_enhancement(
        batch["imgl"], batch["imgr"], types=["while", "max"]
    )

    for iter in range(n_disp_pr):
        adjusted_loss_gamma = args.loss_gamma ** (15 / (n_disp_pr - 1))
        iter_weight = adjusted_loss_gamma ** (
            n_disp_pr - iter - 1
        )  # if iter > 0 else adjusted_loss_gamma**(n_disp_pr - 2)
        loss += iter_weight * unsupervised_stereo_loss(
            imgl_enc,
            imgr_enc,
            output["displ_list"][iter],
            output["dispr_list"][iter],
            args.loss_weights,
            args.border,
            writer if iter == n_disp_pr - 1 else None,
        )

    if "imgs_pr" in output:
        loss += unsupervised_sonar_loss(
            batch["imgs"],
            output["imgs_pr"],
            args.loss_weights,
            output["imgs_mask"],
            writer if iter == n_disp_pr - 1 else None,
        )
    return loss
