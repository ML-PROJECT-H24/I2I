

import torch
from torch import nn
from torch.nn import functional as F

class VAEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VAEResidualBlock, self).__init__()

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # (batch_size, in_channels, height, width) -> (batch_size, in_channels, height, width)
        x = self.groupnorm_1(x)
        # (batch_size, in_channels, height, width) -> (batch_size, in_channels, height, width)
        x = F.silu(x)
        # (batch_size, in_channels, height, width) -> (batch_size, out_channels, height, width)
        x = self.conv_1(x)
        # (batch_size, out_channels, height, width) -> (batch_size, out_channels, height, width)
        x = self.groupnorm_2(x)
        # (batch_size, out_channels, height, width) -> (batch_size, out_channels, height, width)
        x = F.silu(x)
        # (batch_size, out_channels, height, width) -> (batch_size, out_channels, height, width)
        x = self.conv_2(x)
        # (batch_size, out_channels, height, width) -> (batch_size, out_channels, height, width)
        x = x + self.residual_layer(residual)

        return x


class VAEEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (batch_size, Channel, height, width) -> (batch_size, 32, height, width)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),

            # (batch_size, 32, height, width) -> (batch_size, 32, height, width)
            VAEResidualBlock(32, 32),
            # (batch_size, 32, height, width) -> (batch_size, 32, height, width)
            VAEResidualBlock(32, 32),
            # (batch_size, 32, height, width) -> (batch_size, 32, height / 2, width / 2)
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0),
            # (batch_size, 32, height / 2, width / 2) -> (batch_size, 64, height / 2, width / 2)
            VAEResidualBlock(32, 64), 
            # (batch_size, 64, height / 2, width / 2) -> (batch_size, 64, height / 2, width / 2)
            VAEResidualBlock(64, 64), 
            # (batch_size, 64, height / 2, width / 2) -> (batch_size, 64, height / 4, width / 4)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0), 
            # (batch_size, 64, height / 4, width / 4) -> (batch_size, 128, height / 4, width / 4)
            VAEResidualBlock(64, 128), 
            # (batch_size, 128, height / 4, width / 4) -> (batch_size, 128, height / 4, width / 4)
            VAEResidualBlock(128, 128),

            # (batch_size, 128, height / 4, width / 4) -> (batch_size, 128, height / 4, width / 4)
            nn.GroupNorm(32, 128), 
            # (batch_size, 128, height / 4, width / 4) -> (batch_size, 128, height / 4, width / 4)
            nn.SiLU(), 

            # (batch_size, 128, height / 4, width / 4) -> (batch_size, 8, height / 4, width / 4). 
            nn.Conv2d(128, 8, kernel_size=3, padding=1), 
            # (batch_size, 8, height / 4, width / 4) -> (batch_size, 8, height / 4, width / 4)
            nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, channel, height, width) -> (batch_size, 8, height / 4, width / 4)
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric (see #8)
                # (batch_size, Channel, height, width) -> (batch_size, Channel, height + 1, width + 1)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (batch_size, 8, height / 4, width / 4) -> 2 * (batch_size, 4, height / 4, width / 4)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        return mean, log_variance
    
class VAEDecoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (batch_size, 4, height / 4, width / 4) -> (batch_size, 4, height / 4, width / 4)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            # (batch_size, 4, height / 8, width / 8) -> (batch_size, 128, height / 4, width / 4)
            nn.Conv2d(4, 128, kernel_size=3, padding=1),

            # (batch_size, 128, height / 4, width / 4) -> (batch_size, 128, height / 4, width / 4)
            VAEResidualBlock(128, 128), 
            # (batch_size, 128, height / 4, width / 4) -> (batch_size, 128, height / 4, width / 4)
            VAEResidualBlock(128, 128), 
            # (batch_size, 128, height / 4, width / 4) -> (batch_size, 128, height / 4, width / 4)
            VAEResidualBlock(128, 128), 
            # (batch_size, 128, height / 4, width / 4) -> (batch_size, 128, height / 2, width / 2)
            nn.Upsample(scale_factor=2), 
            # (batch_size, 128, height / 2, width / 2) -> (batch_size, 128, height / 2, width / 2)
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            # (batch_size, 128, height / 2, width / 2) -> (batch_size, 64, height / 2, width / 2)
            VAEResidualBlock(128, 64), 
            # (batch_size, 64, height / 2, width / 2) -> (batch_size, 64, height / 2, width / 2)
            VAEResidualBlock(64, 64), 
            # (batch_size, 64, height / 2, width / 2) -> (batch_size, 64, height / 2, width / 2)
            VAEResidualBlock(64, 64), 
            # (batch_size, 64, height / 2, width / 2) -> (batch_size, 64, height, width)
            nn.Upsample(scale_factor=2), 
            # (batch_size, 64, height, width) -> (batch_size, 64, height, width)
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            # (batch_size, 64, height, width) -> (batch_size, 32, height, width)
            VAEResidualBlock(64, 32), 
            # (batch_size, 32, height, width) -> (batch_size, 32, height, width)
            VAEResidualBlock(32, 32), 
            # (batch_size, 32, height, width) -> (batch_size, 32, height, width)
            VAEResidualBlock(32, 32), 

            # (batch_size, 32, height, width) -> (batch_size, 32, height, width)
            nn.GroupNorm(32, 32), 
            # (batch_size, 32, height, width) -> (batch_size, 32, height, width)
            nn.SiLU(), 
            # (batch_size, 32, height, width) -> (batch_size, 3, height, width)
            nn.Conv2d(32, 3, kernel_size=3, padding=1), 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, 4, height / 8, width / 8) -> (batch_size, 3, height, width)
        for module in self:
            x = module(x)
        return x





        

            

        

