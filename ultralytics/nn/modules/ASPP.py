import torch
from torch import nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """空洞空间金字塔池化模块（适配YOLOv8参数规范）"""
    def __init__(self, c1, c2=256, rates=(6, 12, 18)):
        """
        Args:
            c1: 输入通道数（由框架自动传递）
            c2: 输出通道数（默认256）
            rates: 空洞卷积的扩张率（默认[6,12,18]）
        """
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 1, bias=False)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=rates[0], dilation=rates[0], bias=False)
        self.conv3 = nn.Conv2d(c1, c2, 3, padding=rates[1], dilation=rates[1], bias=False)
        self.conv4 = nn.Conv2d(c1, c2, 3, padding=rates[2], dilation=rates[2], bias=False)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c2, 1, bias=False)
        )
        self.project = nn.Sequential(
            nn.Conv2d(5 * c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        # 保持原有前向逻辑不变
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = F.interpolate(self.global_pool(x), size=x.shape[2:], mode='bilinear', align_corners=True)
        return self.project(torch.cat([x1, x2, x3, x4, x5], dim=1))