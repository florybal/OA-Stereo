import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append(".")

from src.networks import *
from src.utils_net import soft_regresstion


class OAStereo(SubModule):
    def __init__(
        self,
        feature_chan=128,
        pyramid_levels=4,
        lookup_radius=4,
        gru_layers=2,
        max_disp=192,
        use_flsvolume=False,
        cat_flsvolume=False,
        flsvolume_size_k=512,
    ):
        super().__init__()

        # constants
        self.down_sample = 8
        self.chan_list = [feature_chan, feature_chan, feature_chan]
        self.pyramid_levels = pyramid_levels
        self.lookup_radius = lookup_radius
        self.gru_layers = gru_layers
        self.use_flsvolume = use_flsvolume
        self.cat_flsvolume = cat_flsvolume
        self.flsvolume_size_k = flsvolume_size_k
        self.max_disp = max_disp

        # feature encoder
        self.color_encoder = HEPP(in_chan=3, out_chan=feature_chan)
        if use_flsvolume or cat_flsvolume:
            self.fls_volume = FlsVolume(flsvolume_size_k)

        # volume
        self.cost_volume = CostVolume(
            chan=self.chan_list[0], maxdisp=max_disp // self.down_sample
        )
        self.corr_volume = CorrBlock1D(chan=self.chan_list[0])
        if cat_flsvolume:
            in_chan = 2 * pyramid_levels
            self.aggregator = Aggregator(in_chan=in_chan)
        else:
            self.aggregator = Aggregator(in_chan=1)

        # hidden, gate offset encoder
        self.hidden_encoder_list = nn.ModuleList()
        self.gateoff_encoder_list = nn.ModuleList()
        for i in range(self.gru_layers):
            in_chan = self.chan_list[i] + (
                2 * max_disp // self.down_sample if i == 0 else 0
            )
            self.hidden_encoder_list.append(HiddenEncoder(in_chan, feature_chan))
            self.gateoff_encoder_list.append(
                GateOffsetEncoder(self.chan_list[i], feature_chan)
            )

        # updater
        corr_chan = pyramid_levels * (2 * lookup_radius + 1)
        if use_flsvolume:
            corr_chan = corr_chan + pyramid_levels * (2 * lookup_radius + 1)
        self.update = BasicMultiUpdateBlock(gru_layers, feature_chan, corr_chan)

        # upsampler
        self.upsample = ConvexUpSample(feature_chan, self.down_sample)

    def forward(
        self,
        imgl: torch.Tensor,
        imgr: torch.Tensor,
        imgs: torch.Tensor,
        baseline: torch.Tensor,
        intri_c: torch.Tensor,
        intri_s: torch.Tensor,
        Tc: torch.Tensor,
        Ts: torch.Tensor,
        iters=3,
        right=False,
        train=False,
    ) -> torch.Tensor:
        # disp list for training
        if train:
            disp_list = []

        # normalize [0, 1] -> [-1, 1]
        imgl = (2 * imgl - 1).contiguous()
        imgr = (2 * imgr - 1).contiguous()

        if right:
            temp = imgl
            imgl = imgr.flip(dims=[-1])
            imgr = temp.flip(dims=[-1])

        # features
        fmapl_list = self.color_encoder(imgl, self.gru_layers)
        fmapr_list = self.color_encoder(imgr)
        fmapl, fmapr = fmapl_list[0], fmapr_list[0]

        # fls volume
        if self.use_flsvolume or self.cat_flsvolume:
            fls_pyramid = self.fls_volume(imgs, self.pyramid_levels)

        # cost aggregation
        cost = self.cost_volume(fmapl, fmapr)
        if self.cat_flsvolume:
            fls = self.fls_volume.create_volume(
                imgl,
                fls_pyramid,
                baseline,
                intri_c,
                intri_s,
                Tc,
                Ts,
                self.max_disp,
                self.down_sample,
            )
            cost_agg = cost.repeat(1, self.pyramid_levels, 1, 1, 1)
            cost_agg = torch.cat([cost_agg, fls], dim=1)
        else:
            cost_agg = cost
        cost_agg = self.aggregator(cost_agg)
        cost, cost_agg = cost.squeeze(1), cost_agg.squeeze(1)

        # init disp
        disp = soft_regresstion(cost_agg)

        # hidden and gate offsets
        hidden_list = []
        gateoff_list = []
        for i in range(self.gru_layers):
            hidden_in = fmapl_list[i]
            if i == 0:
                hidden_in = torch.cat(
                    [hidden_in, cost.detach(), cost_agg.detach()], dim=1
                )
            hidden_list.append(self.hidden_encoder_list[i](hidden_in))
            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning as gate offset
            gateoff_list.append(self.gateoff_encoder_list[i](fmapl_list[i]))
        self.update.set_gate_offset(gateoff_list)

        # build correlatoin
        corr_pyramid = self.corr_volume(fmapl, fmapr, self.pyramid_levels)

        # updates
        for i in range(iters):
            if train:  # upsample and append
                disp_up = self.upsample(hidden_list[0], disp)
                disp_up = F.interpolate(
                    disp_up, imgl.shape[-2:], mode="bilinear", align_corners=True
                )
                disp_list.append(disp_up)

            if self.gru_layers == 3:  # Update low-res GRU
                hidden_list = self.update(
                    hidden_list, iter32=True, iter16=False, iter08=False, update=False
                )

            if self.gru_layers >= 2:  # Update low-res GRU and mid-res GRU
                iter32 = True if self.gru_layers == 3 else False
                hidden_list = self.update(
                    hidden_list, iter32=iter32, iter16=True, iter08=False, update=False
                )

            iter32 = True if self.gru_layers == 3 else False
            iter16 = True if self.gru_layers >= 2 else False
            corr = self.corr_volume.lookup(corr_pyramid, disp, self.lookup_radius)
            if self.use_flsvolume:
                fls = self.fls_volume.lookup(
                    fls_pyramid,
                    disp.flip(dims=[-1]) if right else disp,
                    baseline,
                    intri_c,
                    intri_s,
                    Tc,
                    Ts,
                    self.down_sample,
                    self.lookup_radius,
                )
                corr = torch.cat([corr, fls.flip(dims=[-1]) if right else fls], dim=1)
            hidden_list, disp_delta = self.update(
                hidden_list,
                corr,
                disp,
                iter32=iter32,
                iter16=iter16,
                iter08=True,
                update=True,
            )

            disp = disp + disp_delta

        # upsample
        disp_up = self.upsample(hidden_list[0], disp)
        disp_up = F.interpolate(
            disp_up, imgl.shape[-2:], mode="bilinear", align_corners=True
        )

        if train:
            disp_list.append(disp_up)
            if right:
                disp_list = [d.flip(dims=[-1]) for d in disp_list]
            return disp_list

        if right:
            disp_up = disp_up.flip(dims=[-1])
        return disp_up


if __name__ == "__main__":
    stereo = (
        OAStereo(feature_chan=128, use_flsvolume=False, cat_flsvolume=True)
        .to("cuda")
        .eval()
    )
    imgl = torch.rand([1, 3, 480, 640]).to("cuda")
    imgr = torch.rand([1, 3, 480, 640]).to("cuda")
    imgs = torch.rand([1, 1, 512, 512]).to("cuda")
    baseline = torch.tensor([0.06], dtype=torch.float32).to("cuda").unsqueeze(0)
    intri_c = (
        torch.tensor([600, 600, 320, 240], dtype=torch.float32).to("cuda").unsqueeze(0)
    )
    intri_s = torch.tensor([130, 20, 5], dtype=torch.float32).to("cuda").unsqueeze(0)
    Tc = (
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
        .to("cuda")
        .unsqueeze(0)
    )
    Ts = (
        torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        .to("cuda")
        .unsqueeze(0)
    )

    # time test
    from utils import gpu_speed_test

    with torch.cuda.amp.autocast(enabled=True):
        gpu_speed_test(
            stereo,
            100,
            "sv_stereo",
            imgl,
            imgr,
            imgs,
            baseline,
            intri_c,
            intri_s,
            Tc,
            Ts,
            6,
        )
