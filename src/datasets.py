import torch
import torch.utils.data as data

from torchvision.transforms import v2
import torchvision.transforms.functional as v2F

import os
import copy
import logging
from glob import glob
from typing import List, Dict

from src.utils import *


class DispAugmentor:
    def __init__(self, crop_size) -> None:
        """class for augmenting stereo images with disparty"""
        ########## camera image ##########
        # color
        self.asymmetric_color_aug_prob = 0.2
        self.color_trans = v2.Compose(
            [
                v2.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=[0, 1.4], hue=0.5
                ),
                v2.RandomGrayscale(p=0.1),
            ]
        )
        # occlusion
        self.eraser_aug_prob = 0.5
        self.eraser_aug_times = 2
        self.erase_bounds = [50, 100]
        # distortion
        self.yjitter_bounds = [-3, 3]
        # crop
        self.crop_size = crop_size
        self.stretch_factor = [0.8, 1.2]

        ########## fls image ##########
        # instensity
        self.instensity_trans = v2.ColorJitter(
            brightness=[0.3, 1.7],
            contrast=[0.6, 1.4],
            saturation=[0, 1.4],
            hue=[-0.5, 0.5],
        )
        self.instensity_bounds = [0.8, 1.2]
        # multipath
        self.multipath_aug_prob = 0.7
        self.multipath_aug_times = 3
        self.range_bounds = [-15, 15]
        self.theta_bounds = [-15, 15]
        self.roll_transparency_bounds = [-0.3, 0.7]
        # noise
        self.noise_aug_prob = 0.9
        self.noise_std = 0.03

    def color_transform(
        self, imgl: torch.Tensor, imgr: torch.Tensor
    ) -> List[torch.Tensor]:
        """Color augmentation"""
        # asymmetric
        if torch.rand(1) < self.asymmetric_color_aug_prob:
            imgl = self.color_trans(imgl)
            imgr = self.color_trans(imgr)
            logging.debug("color augmentation: asymetric color transformation")
        # symmetric
        else:
            w = imgl.shape[-1]
            image_stack = torch.cat([imgl, imgr], dim=-1)
            image_stack = self.color_trans(image_stack)
            imgl, imgr = torch.split(image_stack, w, dim=-1)
            logging.debug("color augmentation: symmetric color transformation")
        return imgl, imgr

    def eraser_transform(
        self, imgl: torch.Tensor, imgr: torch.Tensor
    ) -> List[torch.Tensor]:
        """Occlusion augmentation"""
        ht, wd = imgl.shape[-2:]
        if torch.rand([]) < self.eraser_aug_prob:
            mean_color = imgr.mean(dim=[-1, -2])
            imgr = imgr.clone()
            for _ in range(torch.randint(1, self.eraser_aug_times + 1, [])):
                dx = torch.randint(self.erase_bounds[0], self.erase_bounds[1], [])
                dy = torch.randint(self.erase_bounds[0], self.erase_bounds[1], [])
                x0 = torch.randint(0, wd - dx, [])
                y0 = torch.randint(0, ht - dy, [])
                imgr[:, y0 : y0 + dy, x0 : x0 + dx] = mean_color[..., None, None]
                logging.debug(
                    f"occlusion augmentation: erase right image at [{y0}:{y0+dy}, {x0}:{x0+dx}]"
                )
        return imgl, imgr

    def yjitter_transform(
        self, imgl: torch.Tensor, imgr: torch.Tensor
    ) -> List[torch.Tensor]:
        """Non-ideal undistortion augmentation"""
        d = torch.randint(self.yjitter_bounds[0], self.yjitter_bounds[1] + 1, [])
        if d > 0:
            # up move
            imgr_clone = imgr.clone()
            imgr_clone[:, :-d] = imgr[:, d:]
            imgr = imgr_clone
            logging.debug(
                f"non-ideal undistortion augmentation: move right image up {d}-px"
            )
        elif d < 0:
            # down move
            d = -d
            imgr_clone = imgr.clone()
            imgr_clone[:, d:] = imgr[:, :-d]
            imgr = imgr_clone
            logging.debug(
                f"non-ideal undistortion augmentation: move right image down {d}-px"
            )
        return imgl, imgr

    def crop_transform(
        self,
        imgl: torch.Tensor,
        imgr: torch.Tensor,
        disp=None,
        valid=None,
        intri=None,
    ) -> List[torch.Tensor]:
        """Stretching and cropping augmentation"""
        # stretching first
        _, H, W = imgl.shape
        h_stretch = (
            torch.rand([]) * (self.stretch_factor[1] - self.stretch_factor[0])
            + self.stretch_factor[0]
        )
        w_stretch = (
            torch.rand([]) * (self.stretch_factor[1] - self.stretch_factor[0])
            + self.stretch_factor[0]
        )
        new_h, new_w = int(H * h_stretch), int(W * w_stretch)
        new_h = max(new_h, self.crop_size[0])
        new_w = max(new_w, self.crop_size[1])
        imgl = v2F.resize(imgl, [new_h, new_w])
        imgr = v2F.resize(imgr, [new_h, new_w])
        h_stretch = new_h / H
        w_stretch = new_w / W

        # then corp
        h_start = torch.randint(0, new_h - self.crop_size[0] + 1, [])
        w_start = torch.randint(0, new_w - self.crop_size[1] + 1, [])
        imgl = v2F.crop(imgl, h_start, w_start, self.crop_size[0], self.crop_size[1])
        imgr = v2F.crop(imgr, h_start, w_start, self.crop_size[0], self.crop_size[1])

        if disp is not None:
            disp = v2F.resize(disp, [new_h, new_w]) * w_stretch
            valid = v2F.resize(valid, [new_h, new_w])
            disp = v2F.crop(
                disp, h_start, w_start, self.crop_size[0], self.crop_size[1]
            )
            valid = v2F.crop(
                valid, h_start, w_start, self.crop_size[0], self.crop_size[1]
            )

        if intri is not None:
            intri[0::2] *= w_stretch  # fx cx
            intri[1::2] *= h_stretch  # fy cy
            intri[2] -= w_start  # cx
            intri[3] -= h_start  # cy

        return imgl, imgr, disp, valid, intri

    def intensity_transform(self, imgs: torch.Tensor):
        """intensity augmentation"""
        imgs = self.instensity_trans(imgs)
        for r in range(imgs.shape[-2]):
            rate = (
                torch.rand([1])
                * (self.instensity_bounds[1] - self.instensity_bounds[0])
                + self.instensity_bounds[0]
            )
            imgs[..., r, :] *= rate
        for t in range(imgs.shape[-1]):
            rate = (
                torch.rand([1])
                * (self.instensity_bounds[1] - self.instensity_bounds[0])
                + self.instensity_bounds[0]
            )
            imgs[..., t] *= rate
        logging.debug("intensity augmentation")
        return imgs

    def multipath_transform(self, imgs: torch.Tensor):
        """multipath effect augmentation"""
        for _ in range(self.multipath_aug_times):
            if torch.rand([]) < self.multipath_aug_prob:
                offset_r = torch.randint(*self.range_bounds, [])  # range offset
                offset_t = torch.randint(*self.theta_bounds, [])  # theta offset
                imgs_roll = torch.roll(imgs, shifts=(offset_r, offset_t), dims=(-2, -1))
                rate = (
                    torch.rand([])
                    * (
                        self.roll_transparency_bounds[1]
                        - self.roll_transparency_bounds[0]
                    )
                    + self.roll_transparency_bounds[0]
                )
                imgs = imgs_roll * rate + imgs
                logging.debug(
                    f"multipath effect augmentation: imgs offset [{offset_r}, {offset_t}]"
                )
        return imgs

    def sonar_noise_transform(self, imgs: torch.Tensor) -> torch.Tensor:
        """Noise augmentation for sonar images"""
        if torch.rand([]) < self.noise_aug_prob:
            noise = torch.randn_like(imgs) * self.noise_std
            imgs = imgs + noise
            logging.debug("noise augmentation: added Gaussian noise")
        return imgs

    def __call__(
        self, data: Dict[str, torch.Tensor], sonar_aug=True
    ) -> List[torch.Tensor]:
        imgl = data["imgl"].clone()
        imgr = data["imgr"].clone()
        imgs = data["imgs"].clone() if "imgs" in data.keys() else None
        disp = data["disp"].clone() if "disp" in data.keys() else None
        valid = data["valid"].clone() if "valid" in data.keys() else None
        intri = data["intri_c"].clone() if "intri_c" in data.keys() else None

        imgl, imgr, disp, valid, intri = self.crop_transform(
            imgl, imgr, disp, valid, intri
        )
        data["imgl"] = imgl.clone().contiguous()
        data["imgr"] = imgr.clone().contiguous()
        if "disp" in data.keys():
            data["disp"] = disp.clone().contiguous()
            data["valid"] = valid.clone().contiguous()
        if "intri_c" in data.keys():
            data["intri_c"] = intri.clone().contiguous()

        imgl, imgr = self.color_transform(imgl, imgr)
        imgl, imgr = self.eraser_transform(imgl, imgr)
        imgl, imgr = self.yjitter_transform(imgl, imgr)
        data["imgl_aug"] = imgl.clone().contiguous()
        data["imgr_aug"] = imgr.clone().contiguous()

        if sonar_aug and imgs is not None:
            imgs = self.multipath_transform(imgs)
            imgs = self.intensity_transform(imgs)
            imgs = self.sonar_noise_transform(imgs)
            imgs = torch.clamp(imgs, 0, 1)
            data["imgs_aug"] = imgs.clone().contiguous()
        return data


class StereoDataset(data.Dataset):
    def __init__(
        self,
        augmentor=None,
        sonar_aug=True,
        disp_reader=None,
        valid_reader=None,
        sonar_reader=None,
        sonar_comp=False,
        sonar_norm=False,
    ):
        super().__init__()
        self.disp_reader = disp_reader
        self.valid_reader = valid_reader
        self.sonar_reader = sonar_reader
        self.augmentor = augmentor
        self.sonar_aug = sonar_aug
        self.sonar_comp = sonar_comp
        self.sonar_norm = sonar_norm
        self.imgl_list = []
        self.imgr_list = []
        self.imgs_list = []
        self.disp_list = []
        self.valid_list = []
        # format transforms
        self.img_trans = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )
        self.disp_trans = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=False)]
        )
        self.valid_trans = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=False)]
        )
        self.sonar_trans = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=False),
                v2.GaussianBlur(15, 1),
            ]
        )

    def __getitem__(self, index: int) -> dict:
        data = {}
        data["index"] = index

        # read image
        try:
            data["imgl"] = self.img_trans(read_generally(self.imgl_list[index]))
            data["imgr"] = self.img_trans(read_generally(self.imgr_list[index]))
            if self.valid_reader is not None:
                valid = self.valid_reader(self.valid_list[index])
            if self.disp_reader is not None:
                data["disp"] = self.disp_reader(self.disp_list[index])
            if self.sonar_reader is not None:
                data["imgs"] = self.sonar_reader(self.imgs_list[index])
        except Exception as e:
            index = 0
            data["index"] = index
            data["imgl"] = self.img_trans(read_generally(self.imgl_list[index]))
            data["imgr"] = self.img_trans(read_generally(self.imgr_list[index]))
            if self.valid_reader is not None:
                valid = self.valid_reader(self.valid_list[index])
            if self.disp_reader is not None:
                data["disp"] = self.disp_reader(self.disp_list[index])
            if self.sonar_reader is not None:
                data["imgs"] = self.sonar_reader(self.imgs_list[index])

        if len(data["imgl"].shape) == 2:  # for gray image
            data["imgl"] = torch.tile(data["imgl"][None, ...], (3, 1, 1))
            data["imgr"] = torch.tile(data["imgr"][None, ...], (3, 1, 1))
        elif data["imgl"].shape[0] > 3:
            data["imgl"] = data["imgl"][:3]
            data["imgr"] = data["imgr"][:3]

        if self.sonar_reader is not None:
            # data['imgs'] = self.sonar_reader(self.imgs_list[index])
            data["imgs"] = torch.from_numpy(data["imgs"])[
                None, ...
            ]  # shape [1, r_size, theta_size]
            data["imgs"] = v2F.resize(
                data["imgs"], size=[512, 512]
            )  # r_size, theta_size = (512, 512)
            data["imgs"] = (
                self.sonar_trans(data["imgs"]) / 255
            )  # shape [1, r_size, theta_size]
            # read parameters
            data["Tc"], data["Tc_r"], data["intri_c"], data["baseline"] = (
                self.get_camera_param(index)
            )
            data["Ts"], data["intri_s"] = self.get_sonar_param(index)
            # sonar distance compensation
            data["imgs_org"] = data["imgs"].clone()
            if self.sonar_comp:
                data["imgs"] = fls_distance_compensation(
                    data["imgs"], data["intri_s"][0]
                )  # normalize
            if self.sonar_norm:
                data["imgs"] = fls_distance_normalization(data["imgs"])

        # read disparty
        if self.disp_reader is not None:
            # data['disp'] = self.disp_reader(self.disp_list[index])
            if isinstance(data["disp"], tuple):
                data["disp"], data["valid"] = data["disp"]
                data["disp"] = torch.from_numpy(data["disp"])[None, ...]
                data["disp"] = self.disp_trans(data["disp"])
                data["valid"] = torch.from_numpy(data["valid"])[None, ...].float()
            else:
                data["disp"] = torch.from_numpy(data["disp"])[None, ...]
                data["disp"] = self.disp_trans(data["disp"])
                data["valid"] = ((data["disp"] < 512) & (data["disp"] > 0)).float()

        # read valid
        if self.valid_reader is not None:
            # valid = self.valid_reader(self.valid_list[index])
            if "valid" in data.keys():
                data["valid"] = self.valid_trans(valid) * data["valid"]
            else:
                data["valid"] = self.valid_trans(valid)

        # data augmentation
        if self.augmentor is not None:
            data = self.augmentor(data, self.sonar_aug)
        for k in ["imgl_aug", "imgr_aug", "imgs_aug"]:
            if not k in data.keys():
                k1 = k.replace("_aug", "")
                if k1 in data.keys():
                    data[k] = data[k1].clone()

        return data

    def __mul__(self, v=1):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.imgl_list = v * copy_of_self.imgl_list
        copy_of_self.imgr_list = v * copy_of_self.imgr_list
        copy_of_self.disp_list = v * copy_of_self.disp_list
        copy_of_self.imgs_list = v * copy_of_self.imgs_list
        copy_of_self.valid_list = v * copy_of_self.valid_list
        return copy_of_self

    def __len__(self) -> int:
        return len(self.imgl_list)

    def get_camera_param(self, index):
        raise NotImplementedError

    def get_sonar_param(self, index):
        raise NotImplementedError


class DaveSonar(StereoDataset):
    def __init__(
        self,
        root="datasets/dave_sonar",
        scenes=["1"],
        train=True,
        read_sonar=True,
        image_size=[384, 512],
    ):
        super().__init__(
            augmentor=DispAugmentor(image_size) if train else None,
            sonar_aug=True,
            disp_reader=read_generally,
            sonar_reader=read_generally if read_sonar else None,
            valid_reader=None,
            sonar_comp=True,
            sonar_norm=False,
        )
        assert os.path.exists(root)
        self.root = root
        original_length = len(self.imgl_list)

        imgl_list = imgr_list = disp_list = imgs_list = []
        for d in scenes:
            d = "scene" + d
            imgl_list = imgl_list + sorted(
                glob(os.path.join(root, d, "cam_left", "*.jpg"))
            )
            imgr_list = imgr_list + sorted(
                glob(os.path.join(root, d, "cam_right", "*.jpg"))
            )
            imgs_list = imgs_list + sorted(
                glob(os.path.join(root, d, "sonar", "*.npy"))
            )
            disp_list = disp_list + sorted(
                glob(os.path.join(root, d, "displ", "*.npy"))
            )

        if train:
            imgl_list = imgl_list[:-400]
            imgr_list = imgr_list[:-400]
            imgs_list = imgs_list[:-400]
            disp_list = disp_list[:-400]
        else:
            imgl_list = imgl_list[-400:]
            imgr_list = imgr_list[-400:]
            imgs_list = imgs_list[-400:]
            disp_list = disp_list[-400:]

        self.imgl_list = self.imgl_list + imgl_list
        self.imgr_list = self.imgr_list + imgr_list
        self.imgs_list = self.imgs_list + imgs_list
        self.disp_list = self.disp_list + disp_list
        self.valid_list = self.valid_list + disp_list  # disp as valid
        logging.info(f"Added {len(self.imgl_list) - original_length} from DAVE")

    def get_camera_param(self, index):
        img_dir = os.path.dirname(self.imgl_list[index])
        yaml_path_l = os.path.join(img_dir, os.pardir, "cam_left.yaml")
        yaml_path_r = yaml_path_l.replace("cam_left", "cam_right")
        return read_stereo_yaml(yaml_path_l, yaml_path_r)

    def get_sonar_param(self, index):
        img_dir = os.path.dirname(self.imgl_list[index])
        yaml_path = os.path.join(img_dir, os.pardir, "sonar.yaml")
        extri, intri = read_fls_yaml(yaml_path)
        return extri, intri
