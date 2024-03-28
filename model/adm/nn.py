"""
Various utilities for neural networks.
"""

import math

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np

from functools import partial


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        # return x * th.sigmoid(x)
        return nn.activation.silu(x)



class GroupNorm32(nn.GroupNorm):
    def __call__(self, x):
        return super().__call__(x.astype(jnp.float32)).astype(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if "padding" not in kwargs:
        kwargs["padding"] = 0
    args = args[1:]

    # if dims == 1:
    #     return nn.Conv1d(*args, **kwargs)
    # elif dims == 2:
    #     return nn.Conv2d(*args, **kwargs)
    # elif dims == 3:
    #     return nn.Conv3d(*args, **kwargs)
    # raise ValueError(f"unsupported dimensions: {dims}")
    return nn.Conv(*args, **kwargs)


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    args = args[1:]
    return nn.Dense(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    # args = args[1:]
    if "kernel_size" in kwargs:
        kernel_size = kwargs["kernel_size"]
        kwargs.pop("kernel_size")
    else:
        kernel_size = args[0]
        breakpoint()
    return partial(nn.avg_pool, window_shape=kernel_size, **kwargs)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def zero_module(module):
    jax.tree_map(lambda x: jnp.zeros_like(x), module)


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    # return GroupNorm32(32, channels)
    return GroupNorm32(32)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D array of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] array of positional embeddings.
    """
    half = dim // 2
    freqs = np.exp(
        -np.log(max_period) * np.arange(start=0, stop=half, dtype=np.float32) / half
    )
    args = timesteps[:, None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding
