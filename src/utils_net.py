import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import Conv2dNormActivation


def soft_regresstion(cost: torch.Tensor) -> torch.Tensor:
    """regress from cost volume to disparty map

    Args:
        cost (torch.Tensor): cost volume with shape [b, d, h, w]

    Returns:
        torch.Tensor: disparty map with shape [b, 1, h, w]
    """
    cost = F.softmax(cost, 1)
    weight = torch.arange(cost.shape[1]).reshape([-1, 1, 1]).to(cost.device)
    disp = (cost * weight).sum(dim=1, keepdim=True)
    return disp


class Deconv2dNormActivation(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size = 3, 
                 stride = 1, 
                 padding = None, 
                 norm_layer = nn.BatchNorm2d, 
                 activation_layer = nn.ReLU):
        """transpose 2d convolution layers with normalization and activation

        Args:
            in_channels (int): 
            out_channels (int): 
            kernel_size (int, optional): Defaults to 3.
            stride (int, optional): Defaults to 1.
            padding (_type_, optional): Defaults to None.
            norm_layer (_type_, optional): Defaults to BatchNorm2d.
            activation_layer (_type_, optional): Defaults to nn.ReLU.
        """
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        if norm_layer is None:
            self.norm = None
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        else:
            self.norm = norm_layer(out_channels)
            self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.active = None if activation_layer is None else activation_layer()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.active is not None:
            x = self.active(x)
        return x
        
        
class Deconv3dNormActivation(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size = 3, 
                 stride = 1, 
                 padding = None, 
                 norm_layer = nn.BatchNorm3d, 
                 activation_layer = nn.ReLU):
        """transpose 3d convolution layers with normalization and activation

        Args:
            in_channels (int):
            out_channels (int):
            kernel_size (int, optional): Defaults to 3.
            stride (int, optional): Defaults to 1.
            padding (_type_, optional): Defaults to None.
            norm_layer (_type_, optional): Defaults to nn.BatchNorm3d.
            activation_layer (_type_, optional): Defaults to nn.ReLU.
        """
        super().__init__()
        if padding is None:
            padding = 'same'
        if norm_layer is None:
            self.norm = None
            self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        else:
            self.norm = norm_layer(out_channels)
            self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.active = None if activation_layer is None else activation_layer()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.active is not None:
            x = self.active(x)
        return x

class ResidualBlock2d(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, norm_layer = nn.BatchNorm2d):
        """residual block for 2d convolution

        Args:
            in_chan (int): 
            out_chan (int): 
            norm_layer (_type_, optional): Defaults to BatchNorm2d.
        """
        super().__init__()
        self.conv = nn.Sequential(
            Conv2dNormActivation(in_chan, out_chan, kernel_size=3, stride=1, padding=1, 
                                 norm_layer=norm_layer),
            Conv2dNormActivation(out_chan, out_chan, kernel_size=3, stride=1, padding=1, 
                                 norm_layer=norm_layer),
        )
        if in_chan == out_chan:
            self.trans = None
        else:    
            self.trans = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0), 
                norm_layer(out_chan)
            )
        
    def forward(self, x):
        y = self.conv(x)
        if self.trans is not None:
            x = self.trans(x)
        return F.relu(x+y)
    

class SubModule(nn.Module):
    def __init__(self):
        """A base class that provides a common initialization function
        """
        super().__init__()
    
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def freeze_bn(self):
        for m in self.modules():
            if (isinstance(m, nn.BatchNorm2d) or 
                isinstance(m, nn.BatchNorm3d) or 
                isinstance(m, nn.SyncBatchNorm)):
                m.eval()
        return self