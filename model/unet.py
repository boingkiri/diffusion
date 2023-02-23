import jax.numpy as jnp
from flax import linen as nn

from typing import Tuple, Union, List

from model.modules import *


class UNet(nn.Module):
    image_channels: int = 3
    n_channels: int = 128
    ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 2)# (1, 2, 4, 4) # (1, 2, 2, 4)
    is_atten: Union[Tuple[bool, ...], List[bool]] = (False, True, False, False) # (False, True, True, True) # (False, False, True, True)
    n_blocks: int = 2
    dropout_rate: float = 0.1
    n_heads: int = 1
    n_groups: int = 8
    learn_sigma: bool = False
    @nn.compact
    def __call__(self, x, t, train):
        t = TimeEmbed(self.n_channels, 4 * self.n_channels)(t)
        x = nn.Conv(self.n_channels, (3, 3))(x)
        # Store Downward output for skip connection
        h = [x]

        n_resolution = len(self.ch_mults)
        for i in range(n_resolution):
            out_channels = self.n_channels * self.ch_mults[i]
            for _ in range(self.n_blocks):
                x = UnetDown(out_channels, self.is_atten[i], dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)
                h.append(x)
            if i < n_resolution - 1:
                out_channels = self.n_channels * self.ch_mults[i+1]
                x = Downsample(out_channels)(x)
        
        x = UnetMiddle(out_channels, dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)

        for i in reversed(range(n_resolution)):
            out_channels = self.n_channels * self.ch_mults[i]
            for _ in range(self.n_blocks):
                s = h.pop()
                x = jnp.concatenate((x, s), axis=-1)
                x = UnetUp(out_channels, self.is_atten[i], dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)
            if i > 0:
                out_channels = self.n_channels * self.ch_mults[i - 1]
                x = Upsample(out_channels)(x)

        x = nn.GroupNorm(self.n_groups)(x)
        x = nn.swish(x)

        out_channels = self.image_channels * 2 if self.learn_sigma else self.image_channels
        x = nn.Conv(out_channels, (3, 3))(x)
        return x