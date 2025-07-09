import torch
import torch.optim as optim
import torch.utils.data as data
from torch.cuda.amp import GradScaler

import torch.distributed as dist
import torch.utils.data.distributed as dataDist
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import logging
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")

from options import Options
from show import create_stereo, show_plt
from valid import valid_dave
from src.datasets import StereoDataset, DaveSonar
from src.utils import *
from src.loss import unsupervised_loss


def create_loader(args):
    """create training data loader"""
    datasets = StereoDataset()
    for d in args.train_sets:
        if "dave_sonar" in d:
            # d = "dave_sonar_1_2_x1"
            sq = d.split("_")[2:-1]  # e.g.: ['1', '2']
            repeat = int(d.split("_x")[-1])  # e.g. 1
            read_sonar = args.cat_flsvolume or args.use_flsvolume or args.use_flsloss
            datasets += (
                DaveSonar(scenes=sq, read_sonar=read_sonar, image_size=args.image_size)
                * repeat
            )

    if args.use_ddp:
        train_sampler = dataDist.DistributedSampler(datasets, shuffle=True)
        batch_sampler = data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True
        )
        train_loader = data.DataLoader(
            datasets,
            batch_sampler=batch_sampler,
            pin_memory=True,
            num_workers=args.loader_thread,
        )
    else:
        train_sampler = None
        train_loader = data.DataLoader(
            datasets,
            args.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            num_workers=args.loader_thread,
            worker_init_fn=worker_init_fn,
        )
    logging.info(f"Training with {len(train_loader):d} batches")
    return train_loader, train_sampler


def create_optimizer(args, model):
    """create optimizer and learning rate scheduler"""
    lr = args.learning_rate
    optimizer = optim.AdamW(
        model.parameters(), lr, weight_decay=args.weight_decay, eps=1e-8
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        lr,
        args.train_step,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )
    return optimizer, scheduler


if __name__ == "__main__":
    args = Options().parse()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )
    fix_seeds(1234)

    if args.use_ddp:
        torch.distributed.init_process_group(backend="nccl")
        args.rank = os.getenv("LOCAL_RANK")

    # log
    if not args.use_ddp or args.rank == "0":
        log_dir = os.path.join("runs", args.name)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)

    # device
    if args.device == "cuda":
        assert torch.cuda.is_available()
        if args.use_ddp:
            args.device = f"cuda:{args.rank}"

    stereo = create_stereo(args, train=True)
    train_loader, train_sampler = create_loader(args)
    optimizer_stereo, scheduler_stereo = create_optimizer(args, stereo)
    scaler = GradScaler(enabled="cuda" in args.device)

    fix_seeds(1234)
    keep_training = True
    fls_grider = None
    loss_vis = True
    epoch = 0
    step = 0
    while keep_training:

        if args.use_ddp:
            train_sampler.set_epoch(epoch)
        epoch += 1

        if not args.use_ddp or args.rank == "0":
            pbar = tqdm(train_loader, file=sys.stdout)
        else:
            pbar = train_loader

        for batch in pbar:
            optimizer_stereo.zero_grad()

            # get data on target device
            for key, ipt in batch.items():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = ipt.to(args.device)
            assert stereo.training
            output = {}
            B, _, H, W = batch["imgl"].shape

            # automatic mixed precision
            device_type = "cuda" if "cuda" in args.device else "cpu"
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                # disp prediction
                padder = InputPadder(batch["imgl"].shape, divis_by=32)
                imgl_pad, imgr_pad = padder.pad(batch["imgl_aug"], batch["imgr_aug"])
                intri_c = (
                    padder.pad_intri(batch["intri_c"])
                    if "intri_c" in batch.keys()
                    else None
                )
                intri_s = batch["intri_s"] if "intri_s" in batch.keys() else None
                Tc = batch["Tc"] if "Tc" in batch.keys() else None
                Tc_r = batch["Tc_r"] if "Tc_r" in batch.keys() else None
                Ts = batch["Ts"] if "Ts" in batch.keys() else None
                imgs = batch["imgs_aug"] if "imgs_aug" in batch.keys() else None
                baseline = batch["baseline"] if "baseline" in batch.keys() else None
                displ_list = stereo(
                    imgl_pad,
                    imgr_pad,
                    imgs,
                    baseline,
                    intri_c,
                    intri_s,
                    Tc,
                    Ts,
                    iters=args.update_iters,
                    right=False,
                    train=True,
                )
                dispr_list = stereo(
                    imgl_pad,
                    imgr_pad,
                    imgs,
                    baseline,
                    intri_c,
                    intri_s,
                    Tc_r,
                    Ts,
                    iters=args.update_iters,
                    right=True,
                    train=True,
                )
                output["displ_list"] = [padder.unpad(d) for d in displ_list]
                output["dispr_list"] = [padder.unpad(d) for d in dispr_list]

                # fls_grid for fls loss
                if args.use_flsloss and step >= args.valid_period - 1:
                    if fls_grider is None:
                        fls_grider = FlsGrid(
                            batch["intri_s"],
                            batch["Tc"],
                            batch["Ts"],
                            args.grid_size_ijk,
                            args.sigma,
                        )
                    output["depth"] = disp2depth(
                        displ_list[-1], batch["intri_c"][:, 0], batch["baseline"]
                    )
                    if len(args.train_sets) <= 1:
                        fls_grid = fls_grider.create_grid(
                            output["depth"], batch["imgl"], batch["intri_c"]
                        )
                    else:
                        fls_grid = fls_grider.create_grid_batch(
                            output["depth"],
                            batch["imgl"],
                            batch["intri_c"],
                            batch["intri_s"],
                            batch["Tc"],
                            batch["Ts"],
                            args.grid_size_ijk,
                            args.sigma,
                        )
                    fls_grid = fls_grid.permute(
                        0, 1, 4, 2, 3
                    )  # [B, C, r, theta, phi] -> [B, C, phi, r, theta]
                    output["imgs_pr"] = fls_grid[:, 0].mean(dim=-3, keepdim=True)
                    output["imgs_mask"] = create_mask(fls_grid[:, 0])

                if loss_vis and (not args.use_ddp or args.rank == "0"):

                    def write_with_step(metrics: dict, mode="scalar"):
                        tensorboard_write_dict(writer, metrics, step + 1, mode)

                    loss_vis = False
                else:
                    write_with_step = None

                loss = unsupervised_loss(batch, output, args, write_with_step)

            # optimize
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer_stereo)
            torch.nn.utils.clip_grad_norm_(stereo.parameters(), 1.0)
            scaler.step(optimizer_stereo)
            scaler.update()
            scheduler_stereo.step()

            step = step + 1

            # metrics
            if not args.use_ddp or args.rank == "0":
                inform = {
                    "loss": loss.item(),
                    "learning_rate": optimizer_stereo.param_groups[0]["lr"],
                }
                tensorboard_write_dict(writer, inform, step)

            # valid
            if (not args.use_ddp or args.rank == "0") and step % args.valid_period == 0:
                optimizer_stereo.zero_grad()
                # get module
                model_stereo = stereo.module if args.use_ddp else stereo

                # save model
                save_path = os.path.join(
                    "checkpoints", f"{step}_{args.name}_stereo.pth"
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model_stereo.state_dict(), save_path)
                logging.info(f"Savedfile {os.path.abspath(save_path)}")

                # data visualize
                W = batch["imgl"].shape[-1]
                fig = show_plt(
                    batch["imgl"][0],
                    batch["imgr"][0],
                    batch["disp"][0] / (W - 1),
                    sonar2color(batch["imgs"][0], batch["intri_s"][0, 1]),
                    batch["imgl_aug"][0],
                    batch["imgr_aug"][0],
                    displ_list[-1][0] / (W - 1),
                    sonar2color(batch["imgs_aug"][0], batch["intri_s"][0, 1]),
                    show=False,
                )
                tensorboard_write_dict(writer, {"train": fig}, step, mode="figure")
                tensorboard_write_dict(writer, {"index": batch["index"][0]}, step)

                # valid
                results = valid_dave(model_stereo, border=30, iters=args.update_iters)
                tensorboard_write_dict(writer, results, step)

                model_stereo.train().freeze_bn()

            # loss vis
            if (
                not args.use_ddp or args.rank == "0"
            ) and step % args.valid_period == args.valid_period - 1:
                loss_vis = True

            if step >= args.train_step:
                keep_training = False
                break
    if args.use_ddp:
        dist.destroy_process_group()
