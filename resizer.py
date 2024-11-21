import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class ResBlock(nn.Module):
    def __init__(self, channel_size: int, negative_slope: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(channel_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(channel_size)
        )

    def forward(self, x):
        return x + self.block(x)


class Resizer(nn.Module):
    def __init__(self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_kernels: int = 16,
        num_resblocks: int = 2,
        negative_slope: float = 0.2,
        interpolate_mode: str = "bilinear",
        resizer_image_size: int = 512,
        image_size: int = 256,
        out_width: int = 256, #224,
        out_height: int = 256, #224,
        ):
        super().__init__()
        self.interpolate_mode = interpolate_mode
        self.out_width = out_width
        self.out_height = out_height
        self.scale_factor = image_size / resizer_image_size

        n = num_kernels
        r = num_resblocks
        slope = negative_slope

        self.module1 = nn.Sequential(
            nn.Conv2d(in_channels, n, kernel_size=7, padding=3),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(n, n, kernel_size=1),
            nn.LeakyReLU(slope, inplace=True),
            nn.BatchNorm2d(n)
        )

        resblocks = []
        for i in range(r):
            resblocks.append(ResBlock(n, slope))
        self.resblocks = nn.Sequential(*resblocks)

        self.module3 = nn.Sequential(
            nn.Conv2d(n, n, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n)
        )

        self.module4 = nn.Conv2d(n, out_channels, kernel_size=7,
                                 padding=3)

        self.interpolate = partial(F.interpolate,
                                   scale_factor=self.scale_factor,
                                   mode=self.interpolate_mode,
                                   align_corners=False,
                                   recompute_scale_factor=False)

    def forward(self, x):
        # Continue with the rest of the Resizer model as before
        if x.dim() == 3:  # 3D input (single image)
            x = x.unsqueeze(0)
        residual = self.interpolate(x)

        out = self.module1(x)
        out_residual = self.interpolate(out)

        out = self.resblocks(out_residual)
        out = self.module3(out)
        out = out + out_residual

        out = self.module4(out)

        out = out + residual

        return out