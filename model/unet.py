import jax
import jax.numpy as jnp
from flax import linen as nn

import math

from typing import Optional, Tuple, Union, List


class TimeEmbedding(nn.Module):
    emb_dim: int

    @nn.compact
    def __call__(self, time):
        inv_freq = 1.0 / (
            10000
            ** (jnp.arange(0, self.emb_dim, 2, dtype=float) / self.emb_dim)
        )
        time = jnp.expand_dims(time, -1)
        pos_enc_a = jnp.sin(jnp.repeat(time, self.emb_dim // 2, axis=-1) * inv_freq)
        pos_enc_b = jnp.cos(jnp.repeat(time, self.emb_dim // 2, axis=-1) * inv_freq)
        pos_enc = jnp.concatenate([pos_enc_a, pos_enc_b], axis=-1)
        return pos_enc

        # return emb

class ResidualBlock(nn.Module):
    out_channels: int
    # n_groups: int = 32
    n_groups: int = 8
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, t, train):
        h = nn.GroupNorm(self.n_groups)(x)
        h = nn.swish(h)
        h = nn.Conv(self.out_channels, (3, 3))(h)

        # Add time embedding value
        # t = nn.silu(t)
        t = nn.swish(t)
        t_emb = nn.Dense(self.out_channels)(t)
        h += t_emb[:, None, None, :]

        h = nn.GroupNorm(self.n_groups)(h)
        h = nn.swish(h)
        h = nn.Dropout(self.dropout_rate, deterministic=not train)(h)
        h = nn.Conv(self.out_channels, (3, 3))(h)

        if x.shape != h.shape:
            short = nn.Conv(self.out_channels, (1, 1))(x)
            # short = nn.Conv(self.out_channels, (3, 3))(x)
        else:
            short = x
        return h + short

class AttentionBlock(nn.Module):
    n_channels: int
    n_heads: int = 1
    n_groups: int = 8
    
    @nn.compact
    def __call__(self, x): # x: b x y c
        scale = self.n_channels ** -0.5

        batch_size, height, width, n_channels = x.shape
        head_channels = n_channels // self.n_heads
        # Projection
        # qkv = nn.Dense(self.n_heads * self.n_channels * 3)(x)
        x = nn.GroupNorm(self.n_groups)(x)
        qkv = nn.Conv(self.n_heads * head_channels * 3, (1, 1), use_bias=False)(x) # qkv: b x y h*c*3
        qkv = qkv.reshape(batch_size, -1, self.n_heads, 3 * head_channels) # b (x y) h c*3

        # Split as query, key, value
        q, k, v = jnp.split(qkv, 3, axis=-1) # q,k,v = b (x y) h c

        # Scale dot product 
        atten = jnp.einsum('bihd,bjhd->bijh', q, k) * scale

        # Softmax
        atten = nn.softmax(atten, axis=2)

        # Multiply by value
        res = jnp.einsum('bijh,bjhd->bihd', atten, v)

        # res = res.reshape(batch_size, -1, self.n_heads * self.n_channels)
        res = res.reshape(batch_size, height, width, self.n_heads * head_channels)
        # res = nn.Dense(n_channels)(res)
        res = nn.Conv(self.n_channels, (1, 1))(res)

        # skip connection
        res += x

        # res = res.transpose(0, 2, 1).reshape(batch_size, n_channels, height, width)
        # res = res.reshape(batch_size, height, width, n_channels)

        return res


class UnetDown(nn.Module):
    out_channels: int
    has_atten: bool
    dropout_rate: float
    
    @nn.compact
    def __call__(self, x, t, train):
        x = ResidualBlock(self.out_channels, dropout_rate=self.dropout_rate)(x, t, train)
        if self.has_atten:
            x = AttentionBlock(self.out_channels)(x)
        return x

class UnetUp(nn.Module):
    out_channels: int
    has_atten: bool
    dropout_rate: float
    
    @nn.compact
    def __call__(self, x, t, train):
        x = ResidualBlock(self.out_channels, dropout_rate=self.dropout_rate)(x, t, train)
        if self.has_atten:
            x = AttentionBlock(self.out_channels)(x)
        return x

class UnetMiddle(nn.Module):
    n_channels: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, t, train):
        x = ResidualBlock(self.n_channels, dropout_rate=self.dropout_rate)(x, t, train)
        x = AttentionBlock(self.n_channels)(x)
        x = ResidualBlock(self.n_channels, dropout_rate=self.dropout_rate)(x, t, train)
        return x


class Upsample(nn.Module):
    n_channels: int

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        scale = 2
        x = jax.image.resize(
            x, 
            shape=(B, H * scale, W * scale, C),
            method="nearest")
        x = nn.Conv(self.n_channels, (3, 3))(x)
        return x

class Downsample(nn.Module):
    n_channels: int

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        x = jnp.reshape(x, (B, H // 2, W // 2, C * 4))
        x = nn.Conv(self.n_channels, (3, 3), (1, 1))(x)
        return x


class UNet(nn.Module):
    image_channels: int = 3
    n_channels: int = 128
    ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 2)# (1, 2, 4, 4) # (1, 2, 2, 4)
    is_atten: Union[Tuple[bool, ...], List[bool]] = (False, True, False, False) # (False, True, True, True) # (False, False, True, True)
    n_blocks: int = 2
    dropout_rate: float = 0.1
    n_heads: int = 1

    @nn.compact
    def __call__(self, x, t, train):
        t = TimeEmbedding(self.n_channels)(t)
        t = nn.Dense(self.n_channels * 4)(t)
        t = nn.swish(t)
        t = nn.Dense(self.n_channels * 4)(t)

        # x = nn.Conv(self.n_channels, (3, 3))(x)
        x = nn.Conv(self.n_channels, (7, 7))(x)
        # Store Downward output for skip connection
        h = [x]

        n_resolution = len(self.ch_mults)
        for i in range(n_resolution):
            out_channels = self.n_channels * self.ch_mults[i]
            for _ in range(self.n_blocks):
                x = UnetDown(out_channels, self.is_atten[i], dropout_rate=self.dropout_rate)(x, t, train)
                h.append(x)
            if i < n_resolution - 1:
                out_channels = self.n_channels * self.ch_mults[i+1]
                x = Downsample(out_channels)(x)
        
        x = UnetMiddle(out_channels, dropout_rate=self.dropout_rate)(x, t, train)

        for i in reversed(range(n_resolution)):
            out_channels = self.n_channels * self.ch_mults[i]
            for _ in range(self.n_blocks):
                s = h.pop()
                x = jnp.concatenate((x, s), axis=-1)
                x = UnetUp(out_channels, self.is_atten[i], dropout_rate=self.dropout_rate)(x, t, train)
            if i > 0:
                out_channels = self.n_channels * self.ch_mults[i - 1]
                x = Upsample(out_channels)(x)

        x = nn.GroupNorm(8)(x)
        x = nn.swish(x)
        x = nn.Conv(self.image_channels, (3, 3))(x)

        return x