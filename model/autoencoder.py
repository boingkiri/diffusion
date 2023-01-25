import flax.linen as nn
import jax
import jax.numpy as jnp

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
        x = self.conv_in(x)
        x = self.unet_middle(x, t, train)
        for module in self.module_list:
            if type(module) is UnetUp:
                x = module(x, t, train)
            elif type(module) is Upsample:
                x = module(x)

        x = self.norm(x)
        x = nn.swish(x)
        x = self.conv_out(x)

        return x


class VectorQuantizer(nn.Module):
    """
    Original github: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
    This model is ported from pytorch to jax, flax.

    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    n_e: int
    e_dim: int
    beta: float
    remap: str = None
    unknown_index: str = "random"
    sane_index_shape: bool = False
    legacy: bool = True

    def setup(self):
        def zero_centered_uniform(key, shape, dtype=jnp.float_):
            scale = 1.0 / self.n_e
            data = jax.random.uniform(key, shape, minval=-scale, maxval=scale)
            return data
        self.embedding = nn.Embed(self.n_e, self.e_dim, embedding_init=zero_centered_uniform)

    def __call__(self, z):
        z_flatten = jnp.reshape(z, (-1, self.e_dim))
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        embedding_value = self.embedding.embedding
        d = jnp.sum(z_flatten ** 2, axis=1, keepdims=True) + \
            jnp.sum(embedding_value ** 2, axis=1) - \
            2 * jnp.einsum('bd, dn -> bn', z_flatten, jnp.transpose(embedding_value))
        min_encoding_indices = jnp.argmin(d, axis=1)
        z_q = self.embedding(min_encoding_indices)
        z_q = jnp.reshape(z_q, z.shape)
        perplexity = None
        min_encodings = None

        loss = self.beta * jnp.mean((jax.lax.stop_gradient(z_q) - z) ** 2) + \
                jnp.mean((z_q - jax.lax.stop_gradient(z)) ** 2)
        
        # Preserve gradients
        z_q = z + jax.lax.stop_gradient(z_q - z)

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)


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
        NotImplementedError("Autoencoder should implement 'encoder' method.")

    def decoder(self, z, train):
        NotImplementedError("Autoencoder should implement 'decoder' method.")
    
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
        return x_rec, posterior


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
    
    def __call__(self, x, train):
        quant, diff, (_, _, ind) = self.encoder(x, train)
        dec = self.decoder(quant, train)
        return dec, diff, ind