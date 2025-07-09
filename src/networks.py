import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation, Conv3dNormActivation

from typing import List

from src.utils import *
from src.utils_net import *


class HEPP(SubModule):
    def __init__(self, in_chan=3, out_chan=64):
        super(HEPP, self).__init__()
        self.conv = nn.Sequential(
            Conv2dNormActivation(
                in_chan, out_chan // 2, kernel_size=3, stride=2, padding=1
            ),
            Conv2dNormActivation(
                out_chan // 2, out_chan // 2, kernel_size=3, stride=1, padding=1
            ),
            Conv2dNormActivation(
                out_chan // 2, out_chan // 2, kernel_size=3, stride=1, padding=1
            ),
            Conv2dNormActivation(
                out_chan // 2, out_chan // 2, kernel_size=3, stride=2, padding=1
            ),
            Conv2dNormActivation(
                out_chan // 2, out_chan // 2, kernel_size=3, stride=1, padding=1
            ),
            Conv2dNormActivation(
                out_chan // 2, out_chan // 2, kernel_size=3, stride=1, padding=1
            ),
            Conv2dNormActivation(
                out_chan // 2, out_chan, kernel_size=3, stride=2, padding=1
            ),
            Conv2dNormActivation(
                out_chan, out_chan, kernel_size=3, stride=1, padding=1
            ),
            Conv2dNormActivation(
                out_chan, out_chan, kernel_size=3, stride=1, padding=1
            ),
        )
        self.weight_init()

    def forward(self, x: torch.Tensor, num_layers=1) -> torch.Tensor:
        x = self.conv(x)
        out_list = [x]
        for _ in range(num_layers - 1):
            x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
            out_list.append(x)
        return out_list


class CostVolume(SubModule):
    def __init__(self, chan: int, maxdisp: int):
        """A class used to build the cost volume

        Args:
            chan (int):
            maxdisp (int): maximum disparity
        """
        super().__init__()
        self.maxdisp = maxdisp
        self.unfold = nn.Unfold((1, maxdisp))
        self.pad = nn.ZeroPad2d((maxdisp - 1, 0, 0, 0))
        self.conv = Conv2dNormActivation(chan, chan, kernel_size=3, stride=1, padding=1)
        self.desc = nn.Conv2d(chan, chan, kernel_size=1, stride=1, padding=0)
        self.weight_init()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # preprocess
        x = self.desc(self.conv(x))
        y = self.desc(self.conv(y))
        # normalize
        x = x / torch.norm(x, 2, 1, True)
        y = y / torch.norm(y, 2, 1, True)
        # cost
        B, C, H, W = x.shape
        y = self.pad(y)  # shape [B, C, H, W+D-1]
        y = self.unfold(y).reshape(B, C, self.maxdisp, H, W)  # shape [B, C, D, H, W]
        x = x.reshape(B, C, 1, H, W)
        cost = (x * y).sum(dim=1, keepdim=True).flip(dims=[2])  # [B, 1, D, H, W]
        return cost


class Aggregator(SubModule):
    def __init__(self, in_chan=1, chan_list=[8, 16, 32, 48]):
        """Implementation of aggregation

        Args:
            chan_list (List[int], optional): hidden channels of 3d convolution.
            Defaults to [8, 16, 32, 48].
        """
        super().__init__()
        self.conv_stem = Conv3dNormActivation(
            in_chan, chan_list[0], kernel_size=3, stride=1, padding=1
        )
        self.conv_down = nn.ModuleList()
        self.conv_up = nn.ModuleList()
        self.conv_skip = nn.ModuleList()
        self.conv_agg = nn.ModuleList()

        for i in range(3):
            # conv down
            self.conv_down.append(
                nn.Sequential(
                    Conv3dNormActivation(
                        chan_list[i],
                        chan_list[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    Conv3dNormActivation(
                        chan_list[i + 1],
                        chan_list[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
            )
            if i == 0:
                self.conv_up.append(
                    nn.ConvTranspose3d(
                        chan_list[i + 1], 1, kernel_size=4, stride=2, padding=1
                    )
                )
            else:
                self.conv_up.append(
                    Deconv3dNormActivation(
                        chan_list[i + 1],
                        chan_list[i],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                )

            if i != 0:  # do not concat the original cost volume
                # conv agg
                self.conv_agg.append(
                    nn.Sequential(
                        Conv3dNormActivation(
                            chan_list[i],
                            chan_list[i],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                        Conv3dNormActivation(
                            chan_list[i],
                            chan_list[i],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    )
                )
                # conv skip
                self.conv_skip.append(
                    Conv3dNormActivation(
                        2 * chan_list[i],
                        chan_list[i],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
        self.weight_init()

    def forward(self, cost: torch.Tensor) -> torch.Tensor:
        # preprocess
        cost = self.conv_stem(cost)
        # downsample
        cost_feats = [cost]
        for i in range(3):
            cost = self.conv_down[i](cost)
            cost_feats.append(cost)
        # upsample, skip and agg
        cost = cost_feats[-1]
        for i in range(3):
            # upsample
            cost = self.conv_up[-i - 1](cost)
            if i != 2:  # do not concat the original cost volume
                # skip
                if cost.shape != cost_feats[-i - 2].shape:
                    cost = F.interpolate(
                        cost, size=cost_feats[-i - 2].shape[-3:], mode="nearest"
                    )
                cost = torch.cat([cost, cost_feats[-i - 2]], 1)
                cost = self.conv_skip[-i - 1](cost)
                # agg
                cost = self.conv_agg[-i - 1](cost)
        return cost  # [b, 1, d, h, w]


class HiddenEncoder(SubModule):
    def __init__(self, in_chan: int, out_chan: int):
        """encoder for hidden states of gru

        Args:
            in_chan (int):
            out_chan (int):
        """
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlock2d(in_chan, out_chan),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.weight_init()

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        hideen = self.conv(fmap)
        return hideen


class GateOffsetEncoder(SubModule):
    def __init__(self, in_chan: int, out_chan: int):
        """encoder for gates' offsets of gru

        Args:
            in_chan (int):
            out_chan (int):
        """
        super().__init__()
        self.out_chan = out_chan
        self.conv = nn.Sequential(
            ResidualBlock2d(in_chan, out_chan),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan * 3, kernel_size=3, padding=1),
        )
        self.weight_init()

    def forward(self, fmap: torch.Tensor) -> List[torch.Tensor]:
        gate_off = self.conv(fmap)
        cz, cr, cq = gate_off.split(split_size=self.out_chan, dim=1)
        return cz, cr, cq


class ConvexUpSample(SubModule):
    def __init__(self, in_chan: int, factor: int):
        """image upsampler with feature's prediction

        Args:
            in_chan (int):
            factor (int):
        """
        super().__init__()
        self.factor = factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (self.factor**2) * 9, 1, padding=0),
        )
        self.weight_init()

    def forward(self, fmap: torch.Tensor, disp_low: torch.Tensor) -> torch.Tensor:
        N, D, H, W = disp_low.shape
        mask = 0.25 * self.conv(fmap)  # scale mask to balence gradients
        mask = mask.reshape(N, 1, 9, self.factor, self.factor, H, W)
        mask = torch.softmax(mask, dim=2)

        disp = F.unfold(self.factor * disp_low, [3, 3], padding=1)
        disp = disp.view(N, D, 9, 1, 1, H, W)
        disp = torch.sum(mask * disp, dim=2)
        disp = disp.permute(0, 1, 4, 2, 5, 3)
        disp = disp.reshape(N, D, self.factor * H, self.factor * W)
        return disp


class FlsVolume(SubModule):
    def __init__(self, flsvolume_size_k):
        # def __init__(self, chan, radius):
        """class for building sonar volume and sampling"""
        super().__init__()
        self.sonar_encoder = HEPP(in_chan=1, out_chan=flsvolume_size_k)
        self.sonar_header = nn.Sequential(
            nn.Conv2d(
                flsvolume_size_k, flsvolume_size_k, kernel_size=1, stride=1, padding=0
            ),
            nn.Sigmoid(),
        )
        self.weight_init()

    def forward(self, imgs: torch.Tensor, num_levels: int) -> torch.Tensor:
        imgs = (2 * imgs - 1).contiguous()
        fmaps = self.sonar_encoder(imgs)[0]
        fls = self.sonar_header(fmaps)  # shape (B, phi, r, theta)
        fls = fls.permute(0, 2, 3, 1)  # shape (B, r, theta, phi)

        fls_pyramid = [fls]
        for _ in range(num_levels - 1):
            fls = F.avg_pool3d(
                fls, kernel_size=2, stride=2
            )  # shape (B, r/2**i, theta/2**i, phi/2**i)
            fls_pyramid.append(fls)
        return fls_pyramid

    @staticmethod
    def _sample_with_grids(
        grids: torch.Tensor,
        fls_pyramid: torch.Tensor,
        intri_c: torch.Tensor,
        intri_s: torch.Tensor,
        Tc: torch.Tensor,
        Ts: torch.Tensor,
        down_sample=8,
    ):
        intri_c_down = intri_c / down_sample
        size_i, size_j, size_k = fls_pyramid[0].shape[-3:]

        # convert to [i, j, k] in fls coordinate
        grids = cam_uvz2xyz(grids, intri_c_down)  # shape [B, D, H, W, 3]
        grids = transform_coordinate(grids, Tc, Ts)
        grids = fls_xyz2ijk(grids, intri_s, [size_i, size_j, size_k])

        # sample
        grid_i = 2 * grids[..., 0] / (size_i - 1) - 1
        grid_j = 2 * grids[..., 1] / (size_j - 1) - 1
        grid_k = 2 * grids[..., 2] / (size_k - 1) - 1
        grids = torch.stack([grid_k, grid_j, grid_i], dim=-1)  # shape [B, D, H, W, 3]

        # pyramid
        out_list = []
        for i in range(len(fls_pyramid)):
            fls = fls_pyramid[i].unsqueeze(
                1
            )  # shape (B, 1, size_i/2**i, size_j/2**i. size_k/2**i)
            fls = F.grid_sample(fls, grids, align_corners=True)  # shape [B, 1, D, H, W]
            out_list.append(fls)

        out = torch.cat(out_list, dim=1)  # shape [B, len(fls_pyramid), D, H, W]
        return out.contiguous()

    @staticmethod
    def create_volume(
        imgl: torch.Tensor,
        fls_pyramid: torch.Tensor,
        baseline: torch.Tensor,
        intri_c: torch.Tensor,
        intri_s: torch.Tensor,
        Tc: torch.Tensor,
        Ts: torch.Tensor,
        max_disp=192,
        down_sample=8,
    ):
        B, _, H, W = imgl.shape
        H_down = int(H / down_sample)
        W_down = int(W / down_sample)
        max_disp_down = int(max_disp / down_sample)

        # create disp volume
        d_range = torch.arange(max_disp_down)
        v_range = torch.arange(H_down)
        u_range = torch.arange(W_down)
        grid_d, grid_v, grid_u = torch.meshgrid(
            [d_range, v_range, u_range], indexing="ij"
        )
        grid_d = (
            grid_d.to(imgl.device).float().unsqueeze(0).expand([B, -1, -1, -1])
        )  # shape [B, D, H, W]
        grid_v = (
            grid_v.to(imgl.device).float().unsqueeze(0).expand([B, -1, -1, -1])
        )  # shape [B, D, H, W]
        grid_u = (
            grid_u.to(imgl.device).float().unsqueeze(0).expand([B, -1, -1, -1])
        )  # shape [B, D, H, W]

        # depth
        intri_c_down = intri_c / down_sample
        grid_z = disp2depth(
            grid_d, intri_c_down[..., 0], baseline
        )  # shape [B, D, H, W]
        grids = torch.stack([grid_u, grid_v, grid_z], dim=-1)  # shape [B, D, H, W, 3]

        return FlsVolume._sample_with_grids(
            grids, fls_pyramid, intri_c, intri_s, Tc, Ts, down_sample
        )

    @staticmethod
    def lookup(
        fls_pyramid: torch.Tensor,
        disp: torch.Tensor,
        baseline: torch.Tensor,
        intri_c: torch.Tensor,
        intri_s: torch.Tensor,
        Tc: torch.Tensor,
        Ts: torch.Tensor,
        down_sample=8,
        radius=4,
    ):
        # create depth volume
        B, _, H, W = disp.shape
        d_range = (
            torch.arange(2 * radius + 1) - radius
        )  # range [-radius, radius], D = 2*radius+1
        # d_range = torch.arange(1) # tensor([0])
        v_range = torch.arange(H)
        u_range = torch.arange(W)
        grid_d, grid_v, grid_u = torch.meshgrid(
            [d_range, v_range, u_range], indexing="ij"
        )
        grid_d = (
            grid_d.to(disp.device).float().unsqueeze(0).expand([B, -1, -1, -1])
        )  # shape [B, D, H, W]
        grid_v = (
            grid_v.to(disp.device).float().unsqueeze(0).expand([B, -1, -1, -1])
        )  # shape [B, D, H, W]
        grid_u = (
            grid_u.to(disp.device).float().unsqueeze(0).expand([B, -1, -1, -1])
        )  # shape [B, D, H, W]
        grid_d = grid_d * 0.05 + disp
        # grid_d = grid_d + disp

        # depth
        intri_c_down = intri_c / down_sample
        grid_z = disp2depth(
            grid_d, intri_c_down[..., 0], baseline
        )  # shape [B, D, H, W]
        grids = torch.stack([grid_u, grid_v, grid_z], dim=-1)  # shape [B, D, H, W, 3]

        fls = FlsVolume._sample_with_grids(
            grids, fls_pyramid, intri_c, intri_s, Tc, Ts, down_sample
        )

        B, L, D, H, W = fls.shape
        return fls.reshape(B, L * D, H, W).contiguous()
        # return fls


class CorrBlock1D(SubModule):
    def __init__(self, chan: int):
        """copy from saft-stereo for building correlation volume"""
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlock2d(chan, chan, nn.InstanceNorm2d),
            nn.Conv2d(chan, chan, kernel_size=3, padding=1),
        )
        self.weight_init()

    def forward(
        self, fmapl: torch.Tensor, fmapr: torch.Tensor, num_levels: int
    ) -> List[torch.Tensor]:
        # conv
        cmapl = self.conv(fmapl)
        cmapr = self.conv(fmapr)
        corr_pyramid = []
        # all pairs correlation
        corr = self._corr(cmapl, cmapr)
        batch, h1, w1, _, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, 1, 1, w2)
        corr_pyramid.append(corr)
        for _ in range(num_levels - 1):
            corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])
            corr_pyramid.append(corr)
        return corr_pyramid

    @staticmethod
    def _corr(fmapl, fmapr):
        B, C, H, W1 = fmapl.shape
        _, _, _, W2 = fmapr.shape
        fmapl = fmapl.view(B, C, H, W1)
        fmapr = fmapr.view(B, C, H, W2)
        corr = torch.einsum("aijk,aijh->ajkh", fmapl, fmapr)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(C).float())

    @staticmethod
    def lookup(corr_pyramid, disp, radius):
        # build coords_x
        B, _, H, W = disp.shape
        coords_x, _ = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing="xy"
        )  # [H, W]
        coords_x = coords_x.float().to(disp.device)  # [H, W]
        coords_x = coords_x[None, None].repeat(B, 1, 1, 1)  # [B, 1, H, W]
        coords_x = coords_x - disp

        out_pyramid = []
        for i in range(len(corr_pyramid)):
            dx = torch.linspace(-radius, radius, 2 * radius + 1)
            dx = dx.view(2 * radius + 1, 1).to(disp.device)  # [2r+1, 1]
            x0 = dx + coords_x.reshape(B * H * W, 1, 1, 1) / 2**i  # [B*H*W, 1, 2r+1, 1]
            y0 = torch.zeros_like(x0)
            coords = torch.cat([x0, y0], dim=-1)  # [B*H*W, 1, 2r+1, 2]
            corr = CorrBlock1D.bilinear_sampler(
                corr_pyramid[i], coords
            )  # [B*H*W, 1, 2r+1]
            corr = corr.view(B, H, W, -1)  # [B, H, W, 2r+1]
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def bilinear_sampler(
        img: torch.Tensor, coords: torch.Tensor, mode="bilinear", mask=False
    ) -> torch.Tensor:
        """Wrapper for grid_sample, uses pixel coordinates

        Args:
            img (torch.Tensor): source
            coords (torch.Tensor):
            mode (str, optional): Defaults to 'bilinear'.
            mask (bool, optional): return valid coords mask. Defaults to False.
        """
        H, W = img.shape[-2:]
        xgrid, ygrid = coords.split([1, 1], dim=-1)
        # normalize to [-1, 1]
        xgrid = 2 * xgrid / (W - 1) - 1
        if H > 1:
            ygrid = 2 * ygrid / (H - 1) - 1
        grid = torch.cat([xgrid, ygrid], dim=-1)
        img = F.grid_sample(img, grid, mode, align_corners=True)
        if mask:
            mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
            return img, mask.float()
        return img


class BasicMotionEncoder(SubModule):
    def __init__(self, in_chan: int, out_chan: int):
        """Encoder for both disp and correlation

        Args:
            in_chan (int):
            out_chan (int):
        """
        super().__init__()
        self.conv_corr = nn.Sequential(
            nn.Conv2d(in_chan, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_disp = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_motion = nn.Sequential(
            nn.Conv2d(64 + 64, out_chan - 1, 3, padding=1), nn.ReLU()
        )
        self.weight_init()

    def forward(self, disp: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        fdisp = self.conv_disp(disp)
        fcorr = self.conv_corr(corr)
        motion = torch.cat([fdisp, fcorr], dim=1)
        motion = self.conv_motion(motion)
        motion = torch.cat([motion, disp], dim=1)
        return motion


class DispartyOffsetPredictor(SubModule):
    def __init__(self, in_chan: int):
        """prediction head for disparty update

        Args:
            in_chan (int):
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
        )
        self.weight_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvGRU(nn.Module):
    def __init__(self, in_chan: int, hidden_chan: int):
        """convolutional GRU

        Args:
            in_chan (int):
            hidden_chan (int):
        """
        super().__init__()
        self.convz = nn.Conv2d(
            hidden_chan + in_chan, hidden_chan, kernel_size=3, stride=1, padding=1
        )
        self.convr = nn.Conv2d(
            hidden_chan + in_chan, hidden_chan, kernel_size=3, stride=1, padding=1
        )
        self.convq = nn.Conv2d(
            hidden_chan + in_chan, hidden_chan, kernel_size=3, stride=1, padding=1
        )
        self.cz = self.cr = self.cq = 0

    def set_gate_offset(self, cz, cr, cq):
        self.cz = cz
        self.cr = cr
        self.cq = cq

    def forward(self, h, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx) + self.cz)
        r = torch.sigmoid(self.convr(hx) + self.cr)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)) + self.cq)
        h = (1 - z) * h + z * q
        return h


class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, gru_layers: int, hidden_chan, motion_in_chan):
        """copy from saft-stereo for disparty updating"""
        super().__init__()
        self.gru_layers = gru_layers
        self.encoder = BasicMotionEncoder(motion_in_chan, hidden_chan)
        self.disp_head = DispartyOffsetPredictor(hidden_chan)

        input_chan = hidden_chan + hidden_chan * (gru_layers >= 2)
        self.gru08 = ConvGRU(input_chan, hidden_chan)

        if gru_layers >= 2:
            input_chan = hidden_chan + hidden_chan * (gru_layers >= 3)
            self.gru16 = ConvGRU(input_chan, hidden_chan)

        if gru_layers >= 3:
            input_chan = hidden_chan + hidden_chan * (gru_layers > 1)
            self.gru32 = ConvGRU(hidden_chan, hidden_chan)

    def set_gate_offset(self, gate_offset_list: List[List[torch.Tensor]]):
        self.gru08.set_gate_offset(*gate_offset_list[0])
        if self.gru_layers >= 2:
            self.gru16.set_gate_offset(*gate_offset_list[1])
        if self.gru_layers >= 3:
            self.gru32.set_gate_offset(*gate_offset_list[2])

    def forward(
        self,
        hidden_list: List[torch.Tensor],
        corr=None,
        disp=None,
        iter08=True,
        iter16=True,
        iter32=True,
        update=True,
    ):
        if iter32:
            addition = F.avg_pool2d(hidden_list[1], 3, stride=2, padding=1)
            hidden_list[2] = self.gru32(hidden_list[2], addition)
        if iter16:
            addition = [F.avg_pool2d(hidden_list[0], 3, stride=2, padding=1)]
            if self.gru_layers >= 3:
                addition.append(
                    F.interpolate(
                        hidden_list[2],
                        hidden_list[1].shape[2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                )
            hidden_list[1] = self.gru16(hidden_list[1], *addition)
        if iter08:
            addition = [self.encoder(disp, corr)]
            if self.gru_layers >= 2:
                addition.append(
                    F.interpolate(
                        hidden_list[1],
                        hidden_list[0].shape[2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                )
            hidden_list[0] = self.gru08(hidden_list[0], *addition)
        if not update:
            return hidden_list

        disp_delta = self.disp_head(hidden_list[0])
        return hidden_list, disp_delta
