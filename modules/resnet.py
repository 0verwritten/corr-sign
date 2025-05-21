from typing import Optional, Callable
import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet18, ResNet as ResNetLib
from torchvision.models.resnet import BasicBlock as BasicBlockLib 

__all__ = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
]
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def checkpoint1(func, *args, **kwargs):
    return func(*args)

class Get_Correlation(nn.Module):
    """
    Memory-reduced replacement for Get_Correlation.
    Input / output tensors untouched: (B,C,T,H,W) → (B,C,T,H,W)
    """
    def __init__(self, channels: int):
        super().__init__()
        red_ch = max(channels // 32, 1)

        # 1×1 reductions
        self.down_q  = nn.Conv3d(channels, red_ch, kernel_size=1, bias=False)
        self.down_kv = nn.Conv3d(channels, channels,  kernel_size=1, bias=False)

        # depth-wise spatial context (same as before)
        self.spa1 = nn.Conv3d(red_ch, red_ch, kernel_size=(9,3,3),
                              padding=(4,1,1), groups=red_ch, bias=False)
        self.spa2 = nn.Conv3d(red_ch, red_ch, kernel_size=(9,3,3),
                              padding=(4,2,2), dilation=(1,2,2),
                              groups=red_ch, bias=False)
        self.spa3 = nn.Conv3d(red_ch, red_ch, kernel_size=(9,3,3),
                              padding=(4,3,3), dilation=(1,3,3),
                              groups=red_ch, bias=False)

        self.weights_s   = nn.Parameter(torch.ones(3) / 3)
        self.weights_dir = nn.Parameter(torch.ones(2) / 2)
        self.back = nn.Conv3d(red_ch, channels, kernel_size=1, bias=False)

    @staticmethod
    def _shift(x, direction: str):
        """
        x : (B,C,T,H,W)
        direction: 'next' or 'prev'
        """
        if direction == 'next':   #   t → t+1   (repeat last frame)
            return torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2)
        else:                     #   t → t-1   (repeat first frame)
            return torch.cat([x[:, :, :1], x[:, :, :-1]], dim=2)

    def forward(self, x):                       # (B,C,T,H,W)
        q = self.down_q(x)                      # (B,C/32,T,H,W)
        kv = self.down_kv(x)                    # (B,C,T,H,W)

        feat_out = 0.0
        for weight, direction in zip(self.weights_dir, ('next', 'prev')):
            v = self._shift(kv, direction)      # (B,C,T,H,W)

            # 1) correlation at *same* spatial position
            sim = (x * v).sum(1, keepdim=True)          # (B,1,T,H,W)

            # 2) gating
            gate = torch.sigmoid(sim) - 0.5             # (B,1,T,H,W)

            # 3) weighted value – broadcast gate over channels
            feat_out = feat_out + weight * v * gate     # (B,C,T,H,W)

        # ── spatial aggregation (unchanged) ──────────────────────────
        x_red = self.down_q(x)                   # (B,C/32,T,H,W)
        agg = ( self.spa1(x_red) * self.weights_s[0]
              + self.spa2(x_red) * self.weights_s[1]
              + self.spa3(x_red) * self.weights_s[2] )
        agg = self.back(agg)                     # (B,C,T,H,W)

        # final modulation
        return feat_out * (torch.sigmoid(agg) - 0.5)
    

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        stride=(1,stride,stride),
        padding=(0,1,1),
        bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm3d(width)
        self.conv2 = conv3x3(width, width, stride=stride)
        self.bn2 = nn.BatchNorm3d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.corr1 = Get_Correlation(self.inplanes)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.corr2 = Get_Correlation(self.inplanes)
        self.alpha = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.corr3 = Get_Correlation(self.inplanes)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
            
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1,stride,stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        N, C, T, H, W = x.size()
        x = self.conv1(x)
        x = checkpoint(self.bn1, x, use_reentrant=False)
        x = self.relu(x)
        x = checkpoint(self.maxpool, x, use_reentrant=False)

        x = checkpoint(self.layer1, x, use_reentrant=False)
        x = checkpoint(self.layer2, x, use_reentrant=False)
        x = x + self.corr1(x) * self.alpha[0]
        x = checkpoint(self.layer3, x, use_reentrant=False)
        x = x + self.corr2(x) * self.alpha[1]
        x = checkpoint1(self.layer4, x, use_reentrant=False)
        x = x + self.corr3(x) * self.alpha[2]
        x = x.transpose(1,2).contiguous()
        x = x.view((-1,)+x.size()[2:]) #bt,c,h,w

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) #bt,c
        x = self.fc(x) #bt,c

        return x

def resnet18(**kwargs):
    """Constructs a ResNet-18 based model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)  
    model.load_state_dict(checkpoint, strict=False)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet34'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)  
    model.load_state_dict(checkpoint, strict=False)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 based model with 3-D kernels (inflated from 2-D weights)."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)          # 50 layers
    checkpoint = model_zoo.load_url(model_urls['resnet50'])
    for ln in list(checkpoint.keys()):
        if 'conv' in ln or 'downsample.0.weight' in ln or ln == 'conv1.weight':
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)        # (O, I, 1, kH, kW)
    model.load_state_dict(checkpoint, strict=False)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 based model with 3-D kernels (inflated from 2-D weights)."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)         # 101 layers
    checkpoint = model_zoo.load_url(model_urls['resnet101'])
    for ln in list(checkpoint.keys()):
        if 'conv' in ln or 'downsample.0.weight' in ln or ln == 'conv1.weight':
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 based model with 3-D kernels (inflated from 2-D weights)."""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)         # 152 layers
    checkpoint = model_zoo.load_url(model_urls['resnet152'])
    modelState = model.state_dict()
    for ln in list(checkpoint.keys()):
        if 'conv' in ln or 'downsample.0.weight' in ln or ln == 'conv1.weight':
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    return model


def resnet18_2d(**kwargs):
    """Constructs a ResNet-18 based model.
    """
    model = ResNetLib(BasicBlockLib, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln]#.unsqueeze(2)  
    model.load_state_dict(checkpoint, strict=False)
    return model


def resnet34_2d(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNetLib(BasicBlockLib, [3, 4, 6, 3], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet34'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln]#.unsqueeze(2)  
    model.load_state_dict(checkpoint, strict=False)
    return model
