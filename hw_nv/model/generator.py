import torch
import torch.nn as nn



class ResBlock(nn.Module):
    def __init__(self, kernel_size, dilations_mat, leaky_slope):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for dilations_vec in dilations_mat:
            sublayers = []
            for dilation in dilations_vec:
                sublayers += [
                    nn.LeakyRelu(leaky_slope),
                    nn.Conv1d(kernel_size, kernel_size, dilation, padding="same")
                ]
            self.layers.append(nn.Sequential(*sublayers))


    def forward(self, input):
        out = input
        for layer in self.layers:
            out = out + layer(out)
        return out


class MRF(nn.Module):
    def __init__(self, kernel_sizes, dilations_mat, leaky_slope):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.layers.append(ResBlock(kernel_size, dilations_mat))

    def forward(self, input):
        out = 0
        for layer in self.layers:
            out += layer[input]
        return out


class Generator(nn.Module):
    def __init__(self, gen_config):
        super().__init__()
        in_c = gen_config["upsample_initial_channel"]
        
        layers = [
            nn.Conv1d(80, gen_config["upsample_initial_channel"], 7, padding="same")
        ]

        for k_u in gen_config["upsample_kernel_sizes"]:
            layers.append(
                nn.LeakyReLU(negative_slope=gen_config["leaky_slope"])
            )
            out_c = in_c // 2
            layers.append( #TODO pad
                nn.ConvTranspose1d(in_c, out_c, k_u, k_u//2, padding=p)
            )
            in_c = out_c
            layers.append(
                MRF(gen_config["resblock_kernel_sizes"], gen_config["resblock_dilation_sizes"], gen_config["leaky_slope"])
            )
        
        layers += [
            nn.LeakyReLU(negative_slope=gen_config["leaky_slope"]),
            nn.Conv1d(in_c, 1, 7, padding="same"),
            nn.Tanh()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)