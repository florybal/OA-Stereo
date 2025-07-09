import torch
from torchvision.transforms import v2

import os
import glob
import logging
import numpy as np
import open3d as o3d
from tqdm import tqdm
from typing import List
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from options import Options
from src.utils import *
from src.oa_stereo import OAStereo


def create_stereo(args, train=True):
    """create stereo network on target device"""
    stereo = OAStereo(
        args.feature_chan,
        args.pyramid_levels,
        args.lookup_radius,
        args.gru_layers,
        args.agg_disp,
        args.use_flsvolume,
        args.cat_flsvolume,
        args.flsvolume_size_k,
    )
    stereo = stereo.to(args.device)

    if args.weight_stereo is not None:
        stereo.load_state_dict(torch.load(args.weight_stereo), strict=False)
    elif args.use_ddp:
        weight_path = "init_weight_stereo.pt"
        if args.rank == "0":
            torch.save(stereo.state_dict(), weight_path)
        torch.distributed.barrier()
        stereo.load_state_dict(torch.load(weight_path, map_location=args.device))
        torch.distributed.barrier()
        if args.rank == "0":
            os.remove(weight_path)

    if args.use_ddp:
        stereo = torch.nn.SyncBatchNorm.convert_sync_batchnorm(stereo)
        stereo = torch.nn.parallel.DistributedDataParallel(
            stereo, device_ids=[int(args.rank)]
        )

    if train:
        stereo = stereo.train()
        if args.use_ddp:
            stereo.module.freeze_bn()
        else:
            stereo.freeze_bn()
        stereo = torch.compile(stereo, disable=(args.no_compile))
    else:
        stereo = stereo.eval()

    return stereo


def pca(features: np.array):
    if features.ndim != 3:
        raise ValueError("shape should be (C, H, W)")
    C, H, W = features.shape

    features_flattened = features.reshape(C, -1).T
    features_1d = PCA(n_components=1).fit_transform(features_flattened)
    return features_1d.T.reshape(1, H, W)


def show_plt(*imgs: List[torch.Tensor], show=True):
    n = len(imgs)
    if n >= 4:
        row = 2
        col = int(np.ceil(n / row))
    else:
        row = 1
        col = n

    fig = plt.figure()
    for i in range(n):
        ax = fig.add_subplot(row, col, i + 1)
        ax.set_title("img%d" % (i + 1))
        img = imgs[i].detach().cpu()

        if img.shape[0] == 1:
            img = img.squeeze(0).numpy().astype("float32")
            img = np.clip(img, 0, 1)
            ax.imshow(img, cmap="jet")
        elif img.shape[0] == 3:
            img = img.permute(1, 2, 0).numpy().astype("float32")
            img = np.clip(img, 0, 1)
            ax.imshow(img)
        else:
            img = pca(img.numpy()).squeeze(0)
            img = np.clip(img, 0, 1)
            ax.imshow(img)
        ax.axis("off")

    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    args = Options().parse()
    stereo = create_stereo(args, train=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    with torch.no_grad():
        imgl_list = sorted(glob.glob(args.left_images, recursive=True))
        imgr_list = sorted(glob.glob(args.right_images, recursive=True))
        imgs_list = sorted(glob.glob(args.sonar_images, recursive=True))
        logging.info(f"Found {len(imgl_list)} images.")

        disp_trans = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])
        img_trans = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        sonar_trans = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=False),
                v2.GaussianBlur(5, 1),
                v2.GaussianBlur(5, 1),
                v2.GaussianBlur(5, 1),
                v2.GaussianBlur(5, 1),
                v2.GaussianBlur(5, 1),
                v2.GaussianBlur(5, 1),
            ]
        )

        Tc, Tc_r, intri_c, baseline = read_stereo_yaml(
            args.cam_left_yaml, args.cam_right_yaml
        )
        Ts, intri_s = read_fls_yaml(args.fls_yaml)
        Ts, intri_s = Ts[None,].to(args.device), intri_s[None,].to(args.device)
        Tc, Tc_r, intri_c, baseline = (
            Tc[None,].to(args.device),
            Tc_r[None,].to(args.device),
            intri_c[None,].to(args.device),
            baseline[None,].to(args.device),
        )

        for i, (imgl_path, imgr_path, imgs_path) in tqdm(
            enumerate(zip(imgl_list, imgr_list, imgs_list))
        ):
            # load image
            imgl = img_trans(read_generally(imgl_path))[None,].to(args.device)
            imgr = img_trans(read_generally(imgr_path))[None,].to(args.device)
            imgs = sonar_trans(read_generally(imgs_path))[None,].to(args.device) / 255.0
            if args.fls_compensation:
                imgs = fls_distance_compensation(imgs, intri_s[0, 0])
            if args.fls_normalization:
                imgs = fls_distance_normalization(imgs)

            # predict
            padder = InputPadder(imgl.shape, divis_by=32)
            imgl_pad, imgr_pad = padder.pad(imgl, imgr)
            intri_pad = padder.pad_intri(intri_c)
            disp = stereo(
                imgl_pad,
                imgr_pad,
                imgs,
                baseline,
                intri_pad,
                intri_s,
                Tc,
                Ts,
                iters=args.update_iters,
                train=False,
            )
            disp = padder.unpad(disp)
            imgl = padder.unpad(imgl_pad)
            imgr = padder.unpad(imgr_pad)

            # show
            H, W = imgl.shape[-2:]
            show_plt(
                imgl[0],
                imgr[0],
                sonar2color(imgs[0], intri_s[0, 1]),
                disp[0, :, :, 50:-50] / (W - 1),
            )
