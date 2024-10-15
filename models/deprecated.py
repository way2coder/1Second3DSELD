# The SELDnet architecture

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from IPython import embed
import sys
import parameters
import torchaudio
from torchvision import models
import torch.nn.init as init
from pytorch_tcn import TCN
from efficientnet_pytorch import EfficientNet

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, learnable: bool = False, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        if learnable:
            # 如果位置编码是可训练的，将 pe 作为 nn.Parameter
            self.pe = nn.Parameter(pe)
        else:
            # 否则，将 pe 注册为缓冲区，不参与训练
            self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    



class CustomResNet18WithSE(nn.Module):
    def __init__(self, in_feat_shape, params):
        super(CustomResNet18WithSE, self).__init__()
        self.params = params
        self._use_se_block = params['use_se_block']

        # 加载预训练的 ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 替换 conv1 层以接受 7 个通道
        self.resnet.conv1 = nn.Conv2d(
            in_feat_shape[1],
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # 修改 layer3 和 layer4 以使用扩张卷积
        self.resnet.layer3 = _make_dilated_layer(self.resnet.layer3, dilation=2)
        self.resnet.layer4 = _make_dilated_layer(self.resnet.layer4, dilation=4)
        
        # 去除 ResNet 的 avgpool 和 fc 层
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # 添加自定义卷积块列表，并在每个 ConvBlock 后添加 SE_Block
        modules = []

        # 第一组 ConvBlock + 可选 SE_Block
        modules.append(ConvBlock(512, 256))
        if self._use_se_block:
            modules.append(SE_Block(256, r=params.get('se_reduction', 16)))
        modules.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))  # 从 [512, 13, 16] 到 [256, 6, 8]
        modules.append(nn.Dropout2d(p=0.3))

        # 第二组 ConvBlock + 可选 SE_Block
        modules.append(ConvBlock(256, 128))
        if self._use_se_block:
            modules.append(SE_Block(128, r=params.get('se_reduction', 16)))
        modules.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))  # 从 [256, 6, 8] 到 [128, 3, 8]
        modules.append(nn.Dropout2d(p=0.4))

        # 第三组 ConvBlock + 可选 SE_Block
        modules.append(ConvBlock(128, 64))
        if self._use_se_block:
            modules.append(SE_Block(64, r=params.get('se_reduction', 16)))
        modules.append(nn.ConvTranspose2d(64, 64, kernel_size=(6, 1), stride=(6, 1)))  # 从 [64, 3, 8] 到 [64, 18, 8]
        modules.append(nn.Upsample(size=(20, 4), mode='bilinear', align_corners=True))  # 从 [64, 18, 8] 到 [64, 20, 4]
        modules.append(nn.Dropout2d(p=0.4))

        # 将模块列表转换为 nn.Sequential
        self.conv_block_list = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.features(x)          # [batch_size, 512, 13, 16]
        x = self.conv_block_list(x)   # [batch_size, 64, 20, 4]
        return x

def _make_dilated_layer(layer, dilation):
    for i, block in enumerate(layer):
        if i == 0:
            # 修改第一个块的卷积层的 stride 为 1
            block.conv1.stride = (1, 1)
            if block.downsample is not None:
                # 修改下采样层的 stride 为 1
                block.downsample[0].stride = (1, 1)
        # 修改第二个卷积层的 dilation 和 padding
        block.conv2.dilation = (dilation, dilation)
        block.conv2.padding = (dilation, dilation)
    return layer


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)
    


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x