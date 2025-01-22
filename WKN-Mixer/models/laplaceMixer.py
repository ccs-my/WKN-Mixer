import numpy as np
import torch
import torch.nn as nn
from layers.Invertible import RevIN
from layers.Projection import ChannelProjection
from utils.decomposition import svd_denoise, NMF

from math import pi
import torch.nn.functional as F

def Laplace(p):
    A = 0.08
    ep = 0.03
    tal = 0.1
    f = 50
    w = 2 * pi * f
    q = torch.tensor(1 - pow(ep, 2))
    y = A * torch.exp((-ep / (torch.sqrt(q))) * (w * (p - tal))) * (-torch.sin(w * (p - tal)))
    return y


class Laplace_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1):

        super(Laplace_fast, self).__init__()

        if in_channels != 1:

            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels)).view(-1, 1)

        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels)).view(-1, 1)

    def forward(self, waveforms):

        time_disc = torch.linspace(0, 1, steps=int((self.kernel_size)))

        p1 = time_disc.cuda() - self.b_.cuda() / self.a_.cuda()

        laplace_filter = Laplace(p1)

        self.filters = (laplace_filter).view(self.out_channels, 1, self.kernel_size).cuda()


        return F.conv1d(waveforms, self.filters, stride=1, padding=1, dilation=1, bias=None, groups=1)


class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        # [B, L, D] or [B, D, L]
        return self.fc2(self.gelu(self.fc1(x)))


class LaplaceLayer(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(LaplaceLayer, self).__init__()
        self.conv = nn.Sequential(
            Laplace_fast(4, 16),
            nn.BatchNorm1d(4),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(144, 72),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(72, out_channel)

    def forward(self, x):
        # (16, 1, 49)
        y = x.clone()
        y = self.conv(y)
        # (16, 4,
        y = y.view(y.size()[0], -1)
        y = self.fc1(y)
        y = self.fc2(y)
        return y


class FactorizedTemporalMixing(nn.Module):
    def __init__(self, input_dim, mlp_dim, sampling) :
        super().__init__()

        assert sampling in [1, 2, 3, 4, 6, 8, 12]
        self.sampling = sampling
        self.temporal_fac = nn.ModuleList([
            MLPBlock(input_dim // sampling, mlp_dim) for _ in range(sampling)
        ])

    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, :, idx::self.sampling] = x_pad

        return y

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            x_samp.append(samp(x[:, :, idx::self.sampling]))

        x = self.merge(x.shape, x_samp)

        return x


class FactorizedChannelMixing(nn.Module):
    def __init__(self, input_dim, factorized_dim):
        super().__init__()

        assert input_dim > factorized_dim
        self.channel_mixing = MLPBlock(input_dim, factorized_dim)

    def forward(self, x):
        return self.channel_mixing(x)


class MixerBlock(nn.Module):
    def __init__(self, tokens_dim, channels_dim, tokens_hidden_dim, channels_hidden_dim, fac_T, fac_C, sampling,
                 norm_flag):
        super().__init__()
        self.tokens_mixing = FactorizedTemporalMixing(tokens_dim, tokens_hidden_dim, sampling,
                                                      channels_dim) if fac_T else MLPBlock(
            tokens_dim, tokens_hidden_dim)
        self.channels_mixing = FactorizedChannelMixing(channels_dim, channels_hidden_dim) if fac_C else None
        self.norm = nn.LayerNorm(channels_dim) if norm_flag else None

    def forward(self, x):
        # token-mixing [B, D, #tokens]
        y = self.norm(x) if self.norm else x
        y = self.tokens_mixing(y.transpose(1, 2)).transpose(1, 2)

        # channel-mixing [B, #tokens, D]
        if self.channels_mixing:
            y += x
            res = y
            y = self.norm(y) if self.norm else y
            y = res + self.channels_mixing(y)

        return y


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.waveLayer = nn.ModuleList([
            LaplaceLayer(1, configs.seq_len) for _ in range(configs.enc_in)
        ])
        self.mlp_blocks = nn.ModuleList([
            MixerBlock(configs.seq_len, configs.enc_in, configs.d_model, configs.d_ff, configs.fac_T, configs.fac_C,
                       configs.sampling, configs.norm) for _ in range(configs.e_layers)
        ])
        self.norm = nn.LayerNorm(configs.enc_in) if configs.norm else None
        self.projection = ChannelProjection(configs.seq_len, configs.pred_len, configs.enc_in, configs.individual)
        # self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        # self.refine = MLPBlock(configs.pred_len, configs.d_model) if configs.refine else None
        self.rev = RevIN(configs.enc_in) if configs.rev else None

        hidden_size = configs.enc_in // 2
        self.hidden_layer = nn.Linear(configs.enc_in, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 3)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x[:, :, 3:]

        x = self.rev(x, 'norm') if self.rev else x

        x = x.transpose(1, 2)
        for idx, layer in enumerate(self.waveLayer):
            x[:, idx, :] = layer(x[:, idx:idx + 1, :])
        x = x.transpose(1, 2)

        for block in self.mlp_blocks:
            x = block(x)

        x = self.norm(x) if self.norm else x
        x = self.projection(x)
        # x = self.refine(x.transpose(1, 2)).transpose(1, 2) if self.refine else x
        x = self.rev(x, 'denorm') if self.rev else x

        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)

        return x
