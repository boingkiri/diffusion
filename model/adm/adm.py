from abc import abstractmethod

import math

import numpy as np
import jax
import flax
import jax.numpy as jnp
import flax.linen as nn

# from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    conv_nd,
    linear,
    avg_pool_nd,
    normalization,
    timestep_embedding,
)

from typing import Optional, Union


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    spacial_dim: int
    embed_dim: int
    num_heads_channels: int
    output_dim: int = None,

    def setup(self):
        # self.positional_embedding = nn.Parameter(
        #     th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        # )
        def normal_initializer(key, shape, dtype=jnp.float32):
            return jax.random.normal(key, shape, dtype=dtype) / self.embed_dim ** 0.5
        self.positional_embedding = self.param("positional_embedding", normal_initializer, (self.spacial_dim ** 2 + 1, self.embed_dim))
        self.qkv_proj = conv_nd(1, self.embed_dim, 3 * self.embed_dim, 1)
        self.c_proj = conv_nd(1, self.embed_dim, self.output_dim or self.embed_dim, 1)
        self.num_heads = self.embed_dim // self.num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def __call__(self, x):
        b, *_spatial, c = x.shape
        x = x.reshape(b, -1, c)  # N(HW)C
        # x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  
        x = jnp.concatenate([jnp.mean(x, axis=-1, keepdims=True), x], axis=-1) # N(HW+1)C
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # N(HW+1)C
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


# class TimestepBlock(nn.Module):
#     """
#     Any module where __call__() takes timestep embeddings as a second argument.
#     """

#     @abstractmethod
#     def __call__(self, x, emb, train):
#         """
#         Apply the module to `x` given `emb` timestep embeddings.
#         """


# class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
class TimestepEmbedSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    # def forward(self, x, emb):
    #     for layer in self:
    #         if isinstance(layer, TimestepBlock):
    #             x = layer(x, emb)
    #         else:
    #             x = layer(x)
    #     return x
    def __call__(self, x, emb, train):
        for layer in self.layers:
            if isinstance(layer, ResBlock):
                x = layer(x, emb, train)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    channels: int
    use_conv: bool
    dims: int = 2
    out_channels: int = None

    def setup(self):
        out_channels = self.out_channels or self.channels
        if self.use_conv:
            self.conv = conv_nd(self.dims, self.channels, out_channels, 3, padding=1)

    def __call__(self, x):
        assert x.shape[-1] == self.channels

        if self.dims == 3:
            x = jax.image.resize(
                x, (x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3] * 2, x.shape[4]), method="nearest"
            )
        else:
            # spatial = x.shape[1:-1] * 2
            spatial = tuple(elem * 2 for elem in x.shape[1:-1])
            x = jax.image.resize(x, (x.shape[0], *spatial, x.shape[-1]), method="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """
    channels: int
    use_conv: bool
    dims: int = 2
    out_channels: int = None

    # def __init__(self, channels, use_conv, dims=2, out_channels=None):
    def setup(self):
        # stride = 2 if self.dims != 3 else (1, 2, 2)
        # stride = 2 if self.dims != 3 else (2, 2, 1)
        if self.dims == 2:
            stride = (2, 2)
        elif self.dims == 3:
            stride = (2, 2, 1)
        else:
            NotImplementedError("Only 2D and 3D are supported.")
        out_channels = self.out_channels or self.channels
        if self.use_conv:
            self.op = conv_nd(
                self.dims, self.channels, self.out_channels, 3, strides=stride, padding=1
            )
        else:
            # assert self.channels == self.out_channels
            assert self.channels == out_channels
            self.op = avg_pool_nd(self.dims, kernel_size=stride, strides=stride)

    def __call__(self, x):
        assert x.shape[-1] == self.channels
        return self.op(x)


# class ResBlock(TimestepBlock):
class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    channels: int
    emb_channels: int
    dropout: float
    out_channels: int = None
    use_conv: bool = False
    use_scale_shift_norm: bool = False
    dims: int = 2
    use_checkpoint: bool = False
    up: bool = False
    down: bool = False

    def setup(self):
        out_channels = self.out_channels or self.channels 
        self.in_layers_pre = nn.Sequential(
            [
                normalization(self.channels),
                nn.activation.silu,
            ]
        )
        # self.in_layers_conv = conv_nd(self.dims, self.channels, self.out_channels, 3, padding=1)
        self.in_layers_conv = conv_nd(self.dims, self.channels, out_channels, 3, padding=1)

        self.updown = self.up or self.down

        if self.up:
            self.h_upd = Upsample(self.channels, False, self.dims)
            self.x_upd = Upsample(self.channels, False, self.dims)
        elif self.down:
            self.h_upd = Downsample(self.channels, False, self.dims)
            self.x_upd = Downsample(self.channels, False, self.dims)
        # else:
        #     self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            [
                nn.activation.silu,
                linear(
                    self.emb_channels,
                    2 * out_channels if self.use_scale_shift_norm else out_channels,
                ),
            ]
        )

        self.out_normalize = normalization(out_channels)
        self.out_activation = nn.activation.silu
        self.out_dropout = nn.Dropout(rate=self.dropout)
        # self.out_conv = conv_nd(self.dims, self.out_channels, self.out_channels, 3, padding=1)
        self.out_conv = conv_nd(self.dims, out_channels, out_channels, 3, padding=1)

        if self.out_channels == self.channels:
            self.skip_connection = None
        elif self.use_conv:
            # self.skip_connection = conv_nd(self.dims, self.channels, self.out_channels, 3, padding=1)
            self.skip_connection = conv_nd(self.dims, self.channels, out_channels, 3, padding=1)
        else:
            # self.skip_connection = conv_nd(self.dims, self.channels, self.out_channels, 1)
            self.skip_connection = conv_nd(self.dims, self.channels, out_channels, 1)

    def __call__(self, x, emb, train=True):
        if self.updown:
            # in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            # h = in_rest(x)
            h = self.in_layers_pre(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            # h = in_conv(h)
            h = self.in_layers_conv(h)
        else:
            # h = self.in_layers(x)
            h = self.in_layers_pre(x)
            h = self.in_layers_conv(h)
        # emb_out = self.emb_layers(emb).type(h.dtype)
        emb_out = self.emb_layers(emb).astype(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            # emb_out = emb_out[..., None]
            emb_out = jnp.expand_dims(emb_out, axis=1)
        if self.use_scale_shift_norm:
            scale, shift = jnp.split(emb_out, 2, axis=-1)
            h = self.out_normalize(h) * (1 + scale) + shift
            h = self.out_activation(h)
            if x.shape[1] <= 16:
                h = self.out_dropout(h, deterministic=not train)
            h = self.out_conv(h)
        else:
            h = h + emb_out
            h = self.out_normalize(h)
            h = self.out_activation(h)
            if x.shape[1] <= 16:
                h = self.out_dropout(h, deterministic=not train)
            h = self.out_conv(h)

        # return self.skip_connection(x) + h
        if self.out_channels == self.channels:
            return x + h
        else:
            return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    channels: int
    num_heads: int = 1
    num_head_channels: int = -1
    use_checkpoint: bool = False
    use_new_attention_order: bool = False

    def setup(self):
        if self.num_head_channels == -1:
            num_heads = self.num_heads
        else:
            assert (
                self.channels % self.num_head_channels == 0
            ), f"q,k,v channels {self.channels} is not divisible by num_head_channels {self.num_head_channels}"
            num_heads = self.channels // self.num_head_channels
        # self.use_checkpoint = use_checkpoint
        self.norm = normalization(self.channels)
        self.qkv = conv_nd(1, self.channels, self.channels * 3, 1)
        if self.use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(num_heads)

        # self.proj_out = zero_module(conv_nd(1, channels, channels, 1))
        self.proj_out = conv_nd(1, self.channels, self.channels, 1, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros)

    # def forward(self, x):
    def __call__(self, x):
        # b, c, *spatial = x.shape
        b, *spatial, c = x.shape
        x = x.reshape(b, -1, c)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        # return (x + h).reshape(b, c, *spatial)
        return (x + h).reshape(b, *spatial, c)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """
    n_heads: int

    @nn.compact
    def __call__(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x T x (H * 3 * C)] tensor of Qs, Ks, and Vs.
        :return: an [N x T x (H * C)] tensor after attention.
        """

        bs, length, width = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        scale = 1 / jnp.sqrt(jnp.sqrt(ch))
        weight = jnp.einsum(
            "btc,bsc->bts",
            (q * scale).view(bs * length, self.n_head, ch),
            (k * scale).view(bs * length, self.n_head, ch),
        )
        weight = jax.nn.softmax(weight.astype(jnp.float32), axis=-1).astype(qkv.dtype)
        a = jnp.einsum("bts,bsc->btc", weight, v.reshape(bs * length, self.n_heads, ch))
        return a.reshape(bs, length, -1)

    # @staticmethod
    # def count_flops(model, _x, y):
    #     return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """
    n_heads: int

    # def forward(self, qkv):
    @nn.compact
    def __call__(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x T x (3 * H * C)] tensor of Qs, Ks, and Vs.
        :return: an [N x T x (H * C)] tensor after attention.
        """
        bs, length, width = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        # q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = jnp.einsum(
            "btc,bsc->bts",
            (q * scale).reshape(bs * self.n_heads, length, ch),
            (k * scale).reshape(bs * self.n_heads, length, ch),
        )  # More stable with f16 than dividing afterwards
        weight = nn.activation.softmax(weight.astype(jnp.float32), axis=-1).astype(qkv.dtype)
        a = jnp.einsum("bts,bsc->btc", weight, v.reshape(bs * self.n_heads, length, ch))
        return a.reshape(bs, length, -1)

    # @staticmethod
    # def count_flops(model, _x, y):
    #     return count_flops_attn(model, _x, y)


class ADMModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    image_size: int
    in_channels: int
    model_channels: int
    out_channels: int
    num_res_blocks: int
    attention_resolutions: int
    dropout: float = 0
    channel_mult: tuple = (1, 2, 4, 8)
    conv_resample: bool = True
    dims: int = 2
    num_classes: Optional[int] = None
    use_checkpoint: bool = False,
    # use_fp16: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    use_new_attention_order: bool =False

    def setup(self):
        if self.num_heads_upsample == -1:
            num_heads_upsample = self.num_heads
        else:
            num_heads_upsample = self.num_head_upsample

        # self.dtype = th.float16 if use_fp16 else th.float32
        # self.dtype = jax.dtypes.bfloat16 if self.use_fp16 else jnp.float32 # TODO: check whether the bfloat16 is valid
        # self.dtype = jnp.bfloat16 if self.use_fp16 else jnp.float32
        
        # self.dtype = jnp.float16

        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            [
                linear(self.model_channels, time_embed_dim),
                nn.activation.silu,
                linear(time_embed_dim, time_embed_dim),
            ]
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embed(self.num_classes, time_embed_dim)

        ch = input_ch = int(self.channel_mult[0] * self.model_channels)

        input_blocks = [TimestepEmbedSequential(
                    [conv_nd(self.dims, self.in_channels, ch, 3, padding=1)]
                )]

        _feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=self.dropout,
                        out_channels=int(mult * self.model_channels),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = int(mult * self.model_channels)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                # input_blocks.append(TimestepEmbedSequential(*layers))
                input_blocks.append(TimestepEmbedSequential(layers))
                _feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                input_blocks.append(
                    TimestepEmbedSequential(
                        [
                            ResBlock(
                                ch,
                                time_embed_dim,
                                self.dropout,
                                out_channels=out_ch,
                                dims=self.dims,
                                use_checkpoint=self.use_checkpoint,
                                use_scale_shift_norm=self.use_scale_shift_norm,
                                down=True,
                            )
                            if self.resblock_updown
                            else Downsample(
                                ch, self.conv_resample, dims=self.dims, out_channels=out_ch
                            )
                        ]
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                _feature_size += ch

        self.input_blocks = input_blocks

        self.middle_block = TimestepEmbedSequential(
            [
                ResBlock(
                    ch,
                    time_embed_dim,
                    self.dropout,
                    dims=self.dims,
                    use_checkpoint=self.use_checkpoint,
                    use_scale_shift_norm=self.use_scale_shift_norm,
                ),
                AttentionBlock(
                    ch,
                    use_checkpoint=self.use_checkpoint,
                    num_heads=self.num_heads,
                    num_head_channels=self.num_head_channels,
                    use_new_attention_order=self.use_new_attention_order,
                ),
                ResBlock(
                    ch,
                    time_embed_dim,
                    self.dropout,
                    dims=self.dims,
                    use_checkpoint=self.use_checkpoint,
                    use_scale_shift_norm=self.use_scale_shift_norm,
                ),
            ]
        )
        _feature_size += ch

        # self.output_blocks = []
        output_blocks = []
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        self.dropout,
                        out_channels=int(self.model_channels * mult),
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = int(self.model_channels * mult)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        )
                    )
                if level and i == self.num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            up=True,
                        )
                        if self.resblock_updown
                        else Upsample(ch, self.conv_resample, dims=self.dims, out_channels=out_ch)
                    )
                    ds //= 2
                # self.output_blocks.append(TimestepEmbedSequential(*layers))
                # output_blocks.append(TimestepEmbedSequential(*layers))
                output_blocks.append(TimestepEmbedSequential(layers))
                _feature_size += ch
        
        self.output_blocks = output_blocks

        self.out = nn.Sequential(
            [
                normalization(ch),
                nn.activation.silu,
                conv_nd(self.dims, input_ch, self.out_channels, 3, padding=1, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros),
            ]
        )

    # def convert_to_fp16(self):
    #     """
    #     Convert the torso of the model to float16.
    #     """
    #     self.input_blocks.apply(convert_module_to_f16)
    #     self.middle_block.apply(convert_module_to_f16)
    #     self.output_blocks.apply(convert_module_to_f16)

    # def convert_to_fp32(self):
    #     """
    #     Convert the torso of the model to float32.
    #     """
    #     self.input_blocks.apply(convert_module_to_f32)
    #     self.middle_block.apply(convert_module_to_f32)
    #     self.output_blocks.apply(convert_module_to_f32)

    def __call__(self, x, timesteps, y=None, train=True):
        """
        Apply the model to an input batch.

        :param x: an [N x ... x C] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x ... x C] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        emb = emb.astype(x.dtype)

        # h = x.astype(self.dtype)
        h = x
        for module in self.input_blocks:
            h = module(h, emb, train=train)
            hs.append(h)
        h = self.middle_block(h, emb, train=train)
        for module in self.output_blocks:
            # h = th.cat([h, hs.pop()], dim=1)
            h = jnp.concatenate([h, hs.pop()], axis=-1)
            h = module(h, emb, train=train)
        # h = h.astype(x.dtype)
        return self.out(h), emb, h


class SuperResModel(ADMModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """
    # image_size: int
    # in_channels: int
    def setup(self):
        self.image_size = self.image_size * 2
        super().setup()
    # def __init__(self, image_size, in_channels, *args, **kwargs):
    #     super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        #  _, new_height, new_width = x.shape
        new_height, new_width, c = x.shape
        upsampled = jax.image.resize(low_res, (new_height, new_width, c), method="bilinear")
        x = jnp.concatenate([x, upsampled], axis=-1)
        return super().__call__(x, timesteps, **kwargs)
