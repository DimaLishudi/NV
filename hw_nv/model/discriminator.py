import torch.nn as nn
import torch.nn.functional as F
from math import ceil


class MPD(nn.Module):
    def __init__(self, p, config, mel_pad):
        """
            Multi-Period sub-Discriminator with period p
        """
        super().__init__()
        self.p = p
        self.mel_pad = mel_pad
        in_c = 1
        out_c = 32

        layers = []

        for _ in range(1, 5):
            layers.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, (5, 1), (3, 1), padding=(2,0)),
                nn.LeakyReLU(negative_slope=config["leaky_slope"])
            ))
            in_c = out_c
            out_c = in_c*2
        
        layers.append(nn.Sequential(
            nn.Conv2d(in_c, in_c, (5, 1), padding=(2, 0)),
            nn.LeakyReLU(negative_slope=config["leaky_slope"]),
        ))
        layers.append(
            nn.Conv2d(in_c, 1, (3,1), padding=(1, 0))
        )
        self.layers = nn.ModuleList(layers)


    def forward(self, input):
        # reshape input
        b, c, t = input.shape
        pad_len = self.p*(ceil(t/self.p)) - t
        input = F.pad(input, (0, pad_len), self.mel_pad)
        input = input.view(b, c, -1, self.p)

        # forwards pass
        fmaps = []
        for layer in self.layers:
            input = layer(input)
            fmaps.append(input)
        input = input.flatten(1)

        return input, fmaps


class MSD(nn.Module):
    def __init__(self, log_p, config):
        """
            Multi-Scale sub-Discriminator with sample rate 2**log_p
        """
        super().__init__()
        self.log_p = log_p  # sampling rate

        # I gave up on making this a loop
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 128, 15, 1, padding=7),
                nn.LeakyReLU(negative_slope=config["leaky_slope"])
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, 41, 2, groups=4, padding=20),
                nn.LeakyReLU(negative_slope=config["leaky_slope"])
            ),
            nn.Sequential(
                nn.Conv1d(128, 256, 41, 2, groups=16, padding=20),
                nn.LeakyReLU(negative_slope=config["leaky_slope"])
            ),
            nn.Sequential(
                nn.Conv1d(256, 512, 41, 4, groups=16, padding=20),
                nn.LeakyReLU(negative_slope=config["leaky_slope"])
            ),
            nn.Sequential(
                nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20),
                nn.LeakyReLU(negative_slope=config["leaky_slope"])
            ),
            nn.Sequential(
                nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20),
                nn.LeakyReLU(negative_slope=config["leaky_slope"])
            ),
            nn.Sequential(
                nn.Conv1d(1024, 1024, 5, 1, padding=2),
                nn.LeakyReLU(negative_slope=config["leaky_slope"])
            ),
            nn.Conv1d(1024, 1, 3, 1, padding=1)
        ])


    def forward(self, input):
        # input downsampling, exact formula to reduce size by 2**log_p
        kernel_size = 4**self.log_p
        stride = 2**self.log_p
        padding = (kernel_size - stride) // 2
        input = F.avg_pool1d(input, kernel_size, stride, padding)

        # forward pass
        fmaps = []
        for layer in self.layers:
            input = layer(input)
            fmaps.append(x)
        input = input.flatten(1)

        return input, fmaps