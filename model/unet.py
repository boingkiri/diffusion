import jax.numpy as jnp
from flax import linen as nn

from typing import Tuple, Union, List

from modules import *


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