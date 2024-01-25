import torch
# import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from model.autoencoder.quantize import VectorQuantizer

# from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from model.autoencoder.distribution import DiagonalGaussianDistribution

from model.autoencoder.torch.module import Normalize, nonlinearity, Upsample, Downsample, ResnetBlock, make_attn
# from model.autoencoder.torch.module 

# from ldm.util import instantiate_from_config

class Downblock(nn.Module):
    block_in: int
    block_out: int
    temb_ch: int
    num_res_blocks: int
    i_level: int
    use_attn: bool
    use_downsample: bool
    resamp_with_conv: bool = True
    dropout: float = 0.0
    attn_type: str = "vanilla"

    def setup(self):
        block_in = self.block_in
        block_out = self.block_out

        block = []
        attn = []
        for i_block in range(self.num_res_blocks):
            block.append(ResnetBlock(in_channels=block_in,
                                     out_channels=block_out,
                                     temb_channels=self.temb_ch,
                                     dropout_rate=self.dropout))
            block_in = block_out
            # if len(attn) > 0:
            if self.use_attn:
                attn.append(make_attn(self.block_out, attn_type=self.attn_type))
        self.block = block
        self.attn = attn
        
        if self.use_downsample:
            self.downsample = Downsample(block_in, self.resamp_with_conv)


class Upblock(nn.Module):
    block_in: int
    block_out: int
    temb_ch: int
    num_res_blocks: int
    i_level: int
    use_attn: bool
    use_upsample: bool
    resamp_with_conv: bool = True
    dropout: float = 0.0
    attn_type: str = "vanilla"

    def setup(self):
        block_in = self.block_in
        block_out = self.block_out

        block = []
        attn = []
        for i_block in range(self.num_res_blocks+1):
            block.append(ResnetBlock(
                in_channels=block_in,
                out_channels=block_out,
                temb_channels=self.temb_ch,
                dropout_rate=self.dropout,
            ))
            block_in = block_out

            if self.use_attn:
                attn.append(make_attn(block_in, attn_type=self.attn_type))
        self.block = block
        self.attn = attn
        if self.i_level != 0:
            self.upsample = Upsample(block_in, self.resamp_with_conv)

class Middleblock(nn.Module):
    block_in: int
    temb_ch: int
    dropout: float = 0.0
    attn_type: str = "vanilla"

    def setup(self):
        self.block_1 = ResnetBlock(in_channels=self.block_in,
                                   out_channels=self.block_in,
                                   temb_channels=self.temb_ch,
                                   dropout_rate=self.dropout)
        self.attn_1 = make_attn(self.block_in, attn_type=self.attn_type)
        self.block_2 = ResnetBlock(in_channels=self.block_in,
                                   out_channels=self.block_in,
                                   temb_channels=self.temb_ch,
                                   dropout_rate=self.dropout)

class Encoder(nn.Module):
    ch: int = 3
    out_ch: int = 3
    ch_mult: tuple = (1,2,4,8)
    num_res_blocks: int = 2
    attn_resolutions: tuple = (16,8)
    dropout: float = 0.0
    resamp_with_conv: bool = True
    in_channels: int = 3
    resolution: int = 256
    z_channels: int = 256
    double_z: bool = True
    use_linear_attn: bool = False
    attn_type: str = "vanilla"

    def setup(self):
        if self.use_linear_attn: 
            attn_type = "linear"
        else:
            attn_type = self.attn_type
        self.temb_ch = 0
        self.num_resolutions = len(self.ch_mult)

        # downsampling
        self.conv_in = nn.Conv(features=self.ch,
                               kernel_size=(3, 3),
                               strides=1,
                               padding=1)

        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(self.ch_mult)
        self.in_ch_mult = in_ch_mult
        # self.down = []
        down_list = []
        for i_level in range(self.num_resolutions):
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * self.ch_mult[i_level]
            down = Downblock(block_in=block_in,
                                block_out=block_out,
                                temb_ch=self.temb_ch,
                                num_res_blocks=self.num_res_blocks,
                                i_level=i_level,
                                use_attn=curr_res in self.attn_resolutions,
                                use_downsample=i_level != self.num_resolutions-1,
                                resamp_with_conv=self.resamp_with_conv,
                                dropout=self.dropout,
                                attn_type=attn_type)
            curr_res = curr_res // 2
            down_list.append(down)
            block_in = block_out
        self.down = down_list

        block_in = block_out

        # middle
        self.mid = Middleblock(block_in=block_in,
                                 temb_ch=self.temb_ch,
                                 dropout=self.dropout,
                                 attn_type=attn_type)
        


        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv(2*self.z_channels if self.double_z else self.z_channels,
                                kernel_size=(3, 3),
                                strides=1,
                                padding=1)

    def __call__(self, x, train):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb, train)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # import pdb; pdb.set_trace()
        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb, train)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb, train)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    ch: int
    out_ch: int
    ch_mult: tuple
    num_res_blocks: int
    attn_resolutions: list
    in_channels: int
    resolution: int
    z_channels: int
    dropout: float = 0.0
    double_z: bool = False
    resamp_with_conv: bool = True
    give_pre_end: bool = False
    tanh_out: bool = False
    use_linear_attn: bool = False
    attn_type: str = "vanilla"

    def setup(self):
        if self.use_linear_attn:
            self.attn_type = "linear"
        self.temb_ch = 0
        self.num_resolutions = len(self.ch_mult)

        self.curr_res = self.resolution // 2 ** (self.num_resolutions-1)
        # self.z_shape = (1, self.z_channels, self.curr_res, self.curr_res)
        self.z_shape = jnp.asarray((1, self.curr_res, self.curr_res, self.z_channels))
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, jnp.prod(self.z_shape)))

        # z to block_in
        self.conv_in = nn.Conv(
            features=self.ch * self.ch_mult[self.num_resolutions-1],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
        )

        # middle
        self.mid = Middleblock(
            block_in=self.ch * self.ch_mult[self.num_resolutions-1],
            temb_ch=self.temb_ch,
            dropout=self.dropout,
            attn_type=self.attn_type
        )

        # upsampling
        # self.up = nn.ModuleList()
        up_list = []
        block_in = self.ch * self.ch_mult[self.num_resolutions-1]
        for i_level in reversed(range(self.num_resolutions)):
            block_out = self.ch * self.ch_mult[i_level]
            up = Upblock(
                block_in=block_in,
                block_out=block_out,
                temb_ch=self.temb_ch,
                num_res_blocks=self.num_res_blocks,
                i_level=i_level,
                use_attn=self.curr_res in self.attn_resolutions,
                use_upsample=i_level != 0,
                resamp_with_conv=self.resamp_with_conv,
                dropout=self.dropout,
                attn_type=self.attn_type
            )
            block_in = block_out
            up_list = [up] + up_list  # prepend to get consistent order
        self.up = up_list

        # end
        self.norm_out = Normalize(block_out)
        self.conv_out = nn.Conv(
            features=self.out_ch,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
        )

    def __call__(self, z, train):
        # assert z.shape[1:] == self.z_shape[1:]
        # self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, train)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb, train)


        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb, train)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)  # You need to define the nonlinearity function
        h = self.conv_out(h)
        if self.tanh_out:
            h = jnp.tanh(h)
        return h

    
class VQModel(nn.Module):
    ddconfig: dict
    lossconfig: dict
    n_embed: int
    embed_dim: int
    ckpt_path: str = None
    image_key: str = "image"
    colorize_nlabels: int = None
    monitor: str = None
    batch_resize_range: tuple = None
    scheduler_config: dict = None
    lr_g_factor: float = 1.0
    remap: dict = None
    sane_index_shape: bool = False
    use_ema: bool = False

    def setup(self):
        self.encoder = Encoder(**self.ddconfig)
        self.decoder = Decoder(**self.ddconfig)
        self.quantize = VectorQuantizer(
            self.n_embed, self.embed_dim, beta=0.25,
            remap=self.remap, sane_index_shape=self.sane_index_shape
        )
        self.quant_conv = nn.Conv(
            features=self.ddconfig["z_channels"],
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
        )
        self.post_quant_conv = nn.Conv(
            features=self.ddconfig["z_channels"],
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
        )
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {self.batch_resize_range}.")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def __call__(self, input, return_pred_indices=False):
        quant, diff, (_, _, ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.colorize = jax.random.normal(jax.random.PRNGKey(0), (3, x.shape[1], 1, 1))
        x = jax.lax.conv(x, filters=self.colorize, strides=(1, 1), padding="VALID")
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x



class VQModelInterface(VQModel):
    # def __init__(self, embed_dim, *args, **kwargs):
    #     super().__init__(embed_dim=embed_dim, *args, **kwargs)
    #     self.embed_dim = embed_dim

    def encode(self, x, train=False):
        h = self.encoder(x, train)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        # return h
        return quant

    def decode(self, h, force_not_quantize=False, train=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, train)
        return dec
    
    def init_apply(self, x, train=False):
        h = self.encode(x, train)
        dec = self.decode(h, train=train)
        return dec
    

    
class AutoencoderKL(nn.Module):
    ddconfig: dict
    lossconfig: dict
    embed_dim: int
    ckpt_path: str = None
    image_key: str = "image"
    colorize_nlabels: int = None
    monitor: str = None

    def setup(self):
        self.encoder = Encoder(**self.ddconfig)
        self.decoder = Decoder(**self.ddconfig)
        assert self.ddconfig["double_z"]
        self.quant_conv = nn.Conv(
            features=2 * self.ddconfig["z_channels"],
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=False,
        )
        self.post_quant_conv = nn.Conv(
            features=self.embed_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=False,
        )

    def encode(self, x, sample_posterior=True):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        dist = DiagonalGaussianDistribution(moments)

        if sample_posterior:
            posterior = dist.sample()
        else:
            posterior = dist.mode()
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def __call__(self, input, sample_posterior=True):
        z = self.encode(input, sample_posterior)
        dec = self.decode(z)
        return dec

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.transpose((0, 3, 1, 2)).astype(jnp.float32)
        return x

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.colorize = jax.random.normal(jax.random.PRNGKey(0), (3, x.shape[1], 1, 1))
        x = jax.lax.conv(x, filters=self.colorize, strides=(1, 1), padding="VALID")
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x


class IdentityFirstStage(nn.Module):
    vq_interface: bool = False

    def encode(self, x):
        return x

    def decode(self, x):
        return x

    def quantize(self, x):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def __call__(self, x):
        return x
