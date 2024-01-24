import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import unfreeze, freeze

from typing import Union, Tuple, List 
from model.modules import UnetUp, UnetMiddle, UnetDown, Downsample, Upsample
from model.autoencoder.distribution import DiagonalGaussianDistribution
from model.autoencoder.quantize import VectorQuantizer


class Encoder(nn.Module):
    image_channels: int = 3
    n_channels: int = 128
    ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 4)
    is_atten: Union[Tuple[bool, ...], List[bool]] = (False, False, False)
    n_blocks: int = 2
    dropout_rate: float = 0.0
    n_heads: int = 1
    n_groups: int= 8
    embed_dim: int= 3

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
        x = nn.Conv(self.embed_dim, (3, 3))(x)

        return x

class Decoder(nn.Module):
    image_channels: int = 3
    n_channels: int = 128
    ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 4)
    is_atten: Union[Tuple[bool, ...], List[bool]] = (False, False, False)
    n_blocks: int = 2
    dropout_rate: float = 0.0
    n_heads: int = 1
    n_groups: int= 8
    embed_dim: int= 3
    
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

    def forward_before_conv_out(self, x, train):
        t = None
        x = self.conv_in(x)
        x = self.unet_middle(x, t, train)
        for module in self.module_list:
            if type(module) is UnetUp:
                x = module(x, t, train)
            elif type(module) is Upsample:
                x = module(x)

        x = self.norm(x)
        x = nn.swish(x)
        return x
    
    def __call__(self, x, train):
        x = self.forward_before_conv_out(x, train)
        x = self.conv_out(x)
        return x



class AbstractAutoEncoder(nn.Module):
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
        pass
    
    def kl_setup(self):
        self.params = {
            "image_channels": self.image_channels,
            "n_channels": self.n_channels,
            "ch_mults": self.ch_mults,
            "is_atten": self.is_atten,
            "n_blocks": self.n_blocks,
            "dropout_rate":self.dropout_rate,
            "n_heads": self.n_heads,
            "n_groups": self.n_groups, 
            "embed_dim": self.embed_dim * 2
        }
    
    def vq_setup(self):
        self.params = {
            "image_channels": self.image_channels,
            "n_channels": self.n_channels,
            "ch_mults": self.ch_mults,
            "is_atten": self.is_atten,
            "n_blocks": self.n_blocks,
            "dropout_rate":self.dropout_rate,
            "n_heads": self.n_heads,
            "n_groups": self.n_groups, 
            "embed_dim": self.embed_dim
        }
    
    def encoder(self, x, train) -> DiagonalGaussianDistribution:
        NotImplementedError("Autoencoder should implement 'encoder' method.")

    def decoder(self, z, train):
        NotImplementedError("Autoencoder should implement 'decoder' method.")
    
    def forward_before_conv_out(self, x , train):
        NotImplementedError("Autoencoder should implement 'forward_before_conv_out' method.")
    
    def __call__(self, x, train):
        NotImplementedError("Autoencoder should implement '__call__' method.")

class AutoEncoderKL(AbstractAutoEncoder):
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
        super().setup()
        super().kl_setup()
        self.encoder_model = Encoder(**self.params)
        self.decoder_model = Decoder(**self.params)
        self.quant_conv = nn.Conv(2 * self.embed_dim, (1, 1))
        self.post_quant_conv = nn.Conv(self.embed_dim, (1, 1))
    
    def encoder(self, x, train) -> DiagonalGaussianDistribution:
        h = self.encoder_model(x, train)
        moments = self.quant_conv(h)
        kl_rng = self.make_rng('gaussian')
        posterior = DiagonalGaussianDistribution(moments, kl_rng)
        return posterior

    def decoder(self, z, train):
        z = self.post_quant_conv(z)
        return self.decoder_model(z, train)
    
    def decoder_before_conv_out(self, z, train):
        z = self.post_quant_conv(z)
        return self.decoder_model.forward_before_conv_out(z, train)

    def forward_before_conv_out(self, x, train, sample_posterior=True):
        posterior = self.encoder(x, train)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        x_rec = self.decoder_before_conv_out(z, train)
        return x_rec, posterior

    def __call__(self, x, train, sample_posterior=True):
        x_rec, posterior = self.forward_before_conv_out(x, train, sample_posterior)
        x_rec_complete = self.decoder_model.conv_out(x_rec)
        return x_rec_complete, x_rec, posterior


class AutoEncoderVQ(AbstractAutoEncoder):
    image_channels: int = 3
    n_channels: int = 128
    ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 4)
    is_atten: Union[Tuple[bool, ...], List[bool]] = (False, False, False)
    n_blocks: int = 2
    dropout_rate: float = 0.1
    n_heads: int = 1
    n_groups: int = 8
    embed_dim: int = 3
    n_embed: int = 8192

    def setup(self):
        super().setup()
        super().vq_setup()
        self.encoder_model = Encoder(**self.params)
        self.decoder_model = Decoder(**self.params)
        self.quant_conv = nn.Conv(self.embed_dim, (1, 1))
        self.post_quant_conv = nn.Conv(self.embed_dim, (1, 1))
        self.quantize = VectorQuantizer(self.n_embed, self.embed_dim, 0.25)
    
    def encoder(self, x, train):
        h = self.encoder_model(x, train)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decoder(self, z, train):
        z = self.post_quant_conv(z)
        dec = self.decoder_model(z, train)
        return dec

    def decoder_before_conv_out(self, z, train):
        z = self.post_quant_conv(z)
        return self.decoder_model.forward_before_conv_out(z, train)

    def forward_before_conv_out(self, x, train):
        quant, codebook_loss, (_, _, ind) = self.encoder(x, train)
        x_rec = self.decoder_before_conv_out(quant, train)
        return x_rec, codebook_loss, ind
    
    def __call__(self, x, train):
        x_rec, codebook_loss, ind = self.forward_before_conv_out(x, train)
        x_rec_complete = self.decoder_model.conv_out(x_rec)
        return x_rec_complete, x_rec, codebook_loss, ind