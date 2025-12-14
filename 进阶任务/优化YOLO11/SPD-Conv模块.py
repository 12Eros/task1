from torch import nn
import torch
class SPDC(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p=None, g=1, d=1, act=True):
        super().__init__()

        # 1. SPD固定下采样倍率
        self.scale = 2
        # 注意：这里告诉模型，这个模块的步幅是2（因为它改变了特征图大小）
        self.stride = self.scale

        # 2. 计算SPD后的通道数 (c1 * 4)
        spd_channels = c1 * (self.scale ** 2)  # 通常是 c1 * 4

        # 3. 处理Padding (自动计算)
        if p is None:
            padding = k // 2 * d
        else:
            padding = p

        self.conv = nn.Conv2d(
            in_channels=spd_channels,
            out_channels=c2,
            kernel_size=k,
            # 关键：卷积的stride必须固定为1，因为下采样已经在SPD阶段完成了
            stride=1,
            padding=padding,
            groups=g,
            dilation=d,
            bias=False
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        B, C, H, W = x.shape

        # 检查尺寸是否能被scale整除
        if H % self.scale != 0 or W % self.scale != 0:
            # 更优雅的处理方式：直接切片到能整除的位置
            H = H - (H % self.scale)
            W = W - (W % self.scale)
            x = x[:, :, :H, :W]

        x = torch.pixel_unshuffle(x, self.scale)

        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x