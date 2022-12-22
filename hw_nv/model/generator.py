import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations_mat, leaky_slope):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for dilations_vec in dilations_mat:
            sublayers = []
            for dilation in dilations_vec:
                sublayers += [
                    nn.LeakyReLU(leaky_slope),
                    nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding="same")
                ]
            self.layers.append(nn.Sequential(*sublayers))


    def forward(self, input):
        for layer in self.layers:
            input= input + layer(input)
        return input


class MRF(nn.Module):
    def __init__(self, channels, kernel_sizes, dilations_mats, leaky_slope):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for kernel_size, dilations_mat in zip(kernel_sizes, dilations_mats):
            self.layers.append(ResBlock(channels, kernel_size, dilations_mat, leaky_slope))

    def forward(self, input):
        out = 0
        for layer in self.layers:
            out += layer(input)
        return out


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_c = config["upsample_initial_channel"]
        
        layers = [
            nn.Conv1d(80, config["upsample_initial_channel"], 7, padding="same")
        ]

        for k in config["upsample_kernel_sizes"]:
            layers.append(
                nn.LeakyReLU(negative_slope=config["leaky_slope"])
            )
            out_c = in_c // 2
            layers.append(
                nn.ConvTranspose1d(in_c, out_c, k, k//2, padding=(k - k//2)//2)
            )
            in_c = out_c
            layers.append(
                MRF(out_c, config["resblock_kernel_sizes"], config["resblock_dilation_sizes"], config["leaky_slope"])
            )
        
        layers += [
            nn.LeakyReLU(negative_slope=config["leaky_slope"]),
            nn.Conv1d(in_c, 1, 7, padding="same"),
            nn.Tanh()
        ]
        self.layers = nn.Sequential(*layers)


    def forward(self, input):
        return self.layers(input)