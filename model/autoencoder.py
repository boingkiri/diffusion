import flax.linen as nn
import jax

from typing import Union, Tuple, List 
from model.modules import UnetUp, UnetMiddle, UnetDown, Downsample, Upsample
from framework.autoencoder.distribution import DiagonalGaussianDistribution


class Encoder(nn.Module):
    image_channels: int = 3
    n_channels: int = 128
    ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 4)
    is_atten: Union[Tuple[bool, ...], List[bool]] = (False, False, False)
    n_blocks: int = 2
    dropout_rate: float = 0.0
    n_heads: int = 1,
    n_groups: int= 8
    
    @nn.compact
    def __call__(self, x, train):
        t = None

        x = nn.Conv(self.n_channels, (3, 3))(x)
        n_resolution = len(self.ch_mults)
        for i in range(n_resolution):
            out_channels = self.n_channels * self.ch_mults[i]
            for _ in range(self.n_blocks):
                x = UnetDown(out_channels, self.is_atten[i], dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)
            if i < n_resolution - 1:
                out_channels = self.n_channels * self.ch_mults[i+1]
                x = Downsample(out_channels)(x)
        
        x = UnetMiddle(out_channels, dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)

        x = nn.GroupNorm(self.n_groups)(x)
        x = nn.swish(x)
        # x = nn.Conv(self.n_channels * self.ch_mults[-1], (3, 3))(x)
        x = nn.Conv(self.image_channels, (3, 3))(x)

        return x

class Decoder(nn.Module):
    image_channels: int = 3
    n_channels: int = 128
    ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 4)
    is_atten: Union[Tuple[bool, ...], List[bool]] = (False, False, False)
    n_blocks: int = 2
    dropout_rate: float = 0.0
    n_heads: int = 1,
    n_groups: int= 8
    
    def setup(self):
        n_resolution = len(self.ch_mults)

        self.conv_in = nn.Conv(self.n_channels, (3, 3))
        self.unet_middle = UnetMiddle(self.n_channels * self.ch_mults[-1], dropout_rate=self.dropout_rate, n_groups=self.n_groups)
        tmp_list = []

        for i in reversed(range(n_resolution)):
            out_channels = self.n_channels * self.ch_mults[i]
            for _ in range(self.n_blocks):
                tmp_list.append(
                    UnetUp(out_channels, self.is_atten[i], dropout_rate=self.dropout_rate, n_groups=self.n_groups))
            if i > 0:
                out_channels = self.n_channels * self.ch_mults[i - 1]
                tmp_list.append(
                    Upsample(out_channels)
                )

        self.module_list = tuple(tmp_list)
        
        self.norm = nn.GroupNorm(self.n_groups)
        self.conv_out = nn.Conv(self.image_channels, (3, 3))

    def get_last_layer(self):
        return self.conv_out

    # @nn.compact
    def __call__(self, x, train):
        t = None
        # n_resolution = len(self.ch_mults)

        # x = nn.Conv(self.n_channels * self.ch_mults[-1], (3, 3))(x)
        # x = nn.Conv(self.n_channels, (3, 3))(x)
        x = self.conv_in(x)
        # x = UnetMiddle(self.n_channels * self.ch_mults[-1], dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)
        x = self.unet_middle(x, t, train)
        # for i in reversed(range(n_resolution)):
        #     out_channels = self.n_channels * self.ch_mults[i]
        #     for _ in range(self.n_blocks):
        #         x = UnetUp(out_channels, self.is_atten[i], dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)
        #     if i > 0:
        #         out_channels = self.n_channels * self.ch_mults[i - 1]
        #         x = Upsample(out_channels)(x)
        # x = self.module_list(x, t=t, train=train)
        for module in self.module_list:
            if type(module) is UnetUp:
                x = module(x, t, train)
            elif type(module) is Upsample:
                x = module(x)

        x = self.norm(x)
        x = nn.swish(x)
        x = self.conv_out(x)

        return x

# class AutoEncoderKL(nn.Module):

class AutoEncoderKL(nn.Module):
    image_channels: int = 3
    n_channels: int = 128
    ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 4)
    is_atten: Union[Tuple[bool, ...], List[bool]] = (False, False, False)
    n_blocks: int = 2
    dropout_rate: float = 0.1
    n_heads: int = 1
    n_groups: int = 8
    embed_dim: int = 3

    def setup(self):
        params = [
            self.image_channels,
            self.n_channels,
            self.ch_mults,
            self.is_atten,
            self.n_blocks,
            self.dropout_rate,
            self.n_heads,
            self.n_groups
        ]
        self.encoder_model = Encoder(*params)
        self.decoder_model = Decoder(*params)
        self.quant_conv = nn.Conv(2 * self.embed_dim, (1, 1))
        self.post_quant_conv = nn.Conv(self.image_channels, (1, 1))
    
    def encoder(self, x, train) -> DiagonalGaussianDistribution:
        h = self.encoder_model(x, train)
        moments = self.quant_conv(h)
        kl_rng = self.make_rng('gaussian')
        posterior = DiagonalGaussianDistribution(moments, kl_rng)
        return posterior

    def decoder(self, z, train):
        z = self.post_quant_conv(z)
        return self.decoder_model(z, train)
    
    def __call__(self, x, train, sample_posterior=True):
        posterior = self.encoder(x, train)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        x_rec = self.decoder(z, train)
        # last_layer = self.decoder_model.variables['params']['Conv_1']['kernel']
        # breakpoint()
        # z = self.encoder_model(x, train)
        # x_rec = self.decoder_model(z, train)
        return x_rec, posterior

    