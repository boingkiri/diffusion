import jax.numpy as jnp
from flax import linen as nn

from typing import Tuple, Union, List, Any

from model.modules import *

from model.unet import UNet


class CustomConv2d(nn.Module):
    in_channels: int
    out_channels: int
    kernel_channels: int
    use_bias: bool = True
    up: bool = False
    down: bool = False
    resample_filter: Union[Tuple[int, ...], List[int]] = (1, 1)
    fused_resample: bool = False
    init_mode: Any = "xavier_normal"
    init_weight: float = 1.0
    init_bias: float = 0.0

    def setup(self):
        init_function = create_initializer(self.init_mode) if type(self.init_mode) is str else self.init_mode
        weight_init_function = lambda key, shape, dtype: init_function(key, shape, dtype) * self.init_weight
        bias_init_function = lambda key, shape, dtype, fan_in, fan_out: init_function(key, shape, dtype, fan_in=fan_in, fan_out=fan_out)*self.init_bias
        self.weight = self.param("W", 
                   weight_init_function, 
                   (self.kernel_channels, self.kernel_channels, self.in_channels, self.out_channels),
                   jnp.float32) \
                    if self.kernel_channels else None
        self.bias = self.param("b", 
                   bias_init_function, 
                   (self.out_channels, ), 
                   jnp.float32,
                   self.kernel_channels*self.kernel_channels*self.in_channels,
                   self.kernel_channels*self.kernel_channels*self.out_channels) \
                   if self.kernel_channels and self.use_bias else None
        f = jnp.asarray(self.resample_filter)
        f = jnp.expand_dims(jnp.outer(f, f), axis=(-1, -2)) / (jnp.sum(f) ** 2)
        # self.resample_filter = f if self.up or self.down else None
        self.resample_filter_outer = f if self.up or self.down else None
        self.dim_spec = ('NHWC', 'HWIO', 'NHWC')

    # @nn.compact
    def __call__(self, x):
        w = self.weight if self.weight is not None else None
        b = self.bias if self.bias is not None else None
        f = self.resample_filter_outer if self.resample_filter_outer is not None else None
        w_pad = w.shape[0] // 2 if w is not None else None
        f_pad = (f.shape[0] - 1) // 2 if f is not None else None

        if self.fused_resample and self.up and w is not None:
            # Conv transpose
            padding = [[max(f_pad - w_pad, 0)] * 2] * 2
            x = jax.lax.conv_general_dilated(x, jnp.tile((f * 4), [1, 1, 1, self.in_channels]), window_strides=(1, 1),
                                             padding=padding, lhs_dilation=(2, 2), rhs_dilation=(1, 1), dimension_numbers=self.dim_spec,
                                             feature_group_count=self.in_channels)
            padding = [[max(w_pad - f_pad, 0)] * 2] * 2
            x = jax.lax.conv_general_dilated(x, w, window_strides=(1, 1), 
                                             padding=padding, dimension_numbers=self.dim_spec)
        elif self.fused_resample and self.down and w is not None:
            padding = [[w_pad+f_pad] * 2] * 2
            x = jax.lax.conv_general_dilated(x, w, window_strides=(1, 1), 
                                             padding=padding, dimension_numbers=self.dim_spec)
            x = jax.lax.conv_general_dilated(x, jnp.tile(f, [1, 1, 1, self.out_channels]), window_strides=(2, 2), 
                                             padding='VALID', dimension_numbers=self.dim_spec, feature_group_count=self.out_channels)
        else:
            if self.up:
                # Conv transpose
                # padding = [[f_pad + 1] * 2] * 2 # TODO: need to be fix, may be.
                padding = [[(f.shape[0] - 1) - f_pad] * 2] * 2
                x = jax.lax.conv_general_dilated(x, jnp.tile((f * 4), [1, 1, 1, self.in_channels]), window_strides=(1, 1),
                                             padding=padding, lhs_dilation=(2, 2), rhs_dilation=None, dimension_numbers=self.dim_spec,
                                             feature_group_count=self.in_channels)
                # x = jax.lax.conv_general_dilated(x, jnp.tile((f * 4), [1, 1, self.in_channels, self.in_channels]), window_strides=(1, 1),
                #                              padding=padding, lhs_dilation=(2, 2), rhs_dilation=None, dimension_numbers=self.dim_spec)
            if self.down:
                padding = [[f_pad] * 2] * 2
                x = jax.lax.conv_general_dilated(x, jnp.tile(f, [1, 1, 1, self.in_channels]), window_strides=(2, 2), 
                                                 padding=padding, dimension_numbers=self.dim_spec, feature_group_count=self.in_channels)
                # x = jax.lax.conv_general_dilated(x, jnp.tile((f * 4), [1, 1, self.in_channels, self.in_channels]), window_strides=(2, 2), 
                #                                  padding=padding, dimension_numbers=self.dim_spec)

            if w is not None:
                padding = [[w_pad] * 2] * 2
                x = jax.lax.conv_general_dilated(x, w, window_strides=(1, 1), 
                                                 padding=padding, dimension_numbers=self.dim_spec)
        
        if b is not None:
            x = x + b.reshape(1, 1, 1, -1)
        return x

class AttentionModule(nn.Module):
    out_channels: int
    num_heads: int
    eps: float

    @nn.compact
    def __call__(self, x):
        init_attn = create_initializer("xavier_attn")
        init_zero = create_initializer("xavier_zero")
        orig_x = x
        x = nn.GroupNorm(epsilon=self.eps)(x)
        qkv = CustomConv2d(self.out_channels, self.out_channels * 3, kernel_channels=1, init_mode=init_attn)(x)
        qkv = qkv.reshape(x.shape[0] * self.num_heads, 3, -1, x.shape[-1] // self.num_heads)
        q, k, v = jnp.split(qkv, 3, axis=1)
        k = k / jnp.sqrt(k.shape[-1])
        
        w = jnp.einsum('bnqc,bnkc->bnqk', q, k)
        w = nn.softmax(w, axis=-1)

        a = jnp.einsum('bnqk,bnkc->bnqc', w, v)
        a = a.reshape(*x.shape)
        x = CustomConv2d(self.out_channels, self.out_channels, kernel_channels=1, init_mode=init_zero)(a) + orig_x
        return x

class UNetBlock(nn.Module):
    in_channels: int
    out_channels: int
    emb_channels: int
    up: bool = False
    down: bool = False
    attention: bool = False
    num_heads: int = None
    channels_per_head = 64
    dropout_rate: float = 0.0
    skip_scale : int = 1
    eps: float = 1e-5
    resample_filter: Union[Tuple[int, ...], List[int]] = (1, 1)
    resample_proj: bool = False
    adaptive_scale: bool = True
    
    def setup(self):
        init = create_initializer("xavier_uniform")
        init_zero = create_initializer("xavier_zero")

        self.norm0 = nn.GroupNorm(epsilon=self.eps)
        self.conv0 = CustomConv2d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_channels=3, 
            up=self.up, down=self.down, 
            resample_filter=self.resample_filter, 
            init_mode=init)
        self.affine = nn.Dense(self.out_channels * (2 if self.adaptive_scale else 1), kernel_init=init)
        self.norm1 = nn.GroupNorm(epsilon=self.eps)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.conv1 = CustomConv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_channels=3, init_mode=init_zero)

        self.skip = None
        if self.out_channels != self.in_channels or self.up or self.down:
            kernel = 1 if self.resample_proj or self.out_channels != self.in_channels else 0
            self.skip = CustomConv2d(
                in_channels=self.in_channels, 
                out_channels=self.out_channels, 
                kernel_channels=kernel, 
                up=self.up, down=self.down,
                resample_filter=self.resample_filter,
                init_mode = init)
        if self.attention:
            self.atten = AttentionModule(self.out_channels, self.num_heads, self.eps)
    
    def __call__(self, x, emb, train):
        orig = x
        x = self.conv0(nn.silu(self.norm0(x)))

        params = self.affine(emb)
        params = jnp.expand_dims(params, axis=(1, 2))

        if self.adaptive_scale:
            scale, shift = params.split(2, axis=-1)
            x = nn.silu(shift + self.norm1(x) * (scale + 1))
        else:
            x = nn.silu(self.norm1(x + params))
        
        x = self.dropout1(x, deterministic=not train)
        x = self.conv1(x)
        skip_orig = self.skip(orig) if self.skip is not None else orig
        x = (x + skip_orig) * self.skip_scale
        if self.attention:
            x = self.atten(x)
            x = x * self.skip_scale
        return x

class UNetpp(nn.Module):
    image_channels: int = 3
    n_channels: int = 128
    label_dim: int = 0
    augment_dim: int = 0
    
    ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 2)# (1, 2, 4, 4) # (1, 2, 2, 4)
    is_atten: Union[Tuple[bool, ...], List[bool]] = (False, True, False, False) # (False, True, True, True) # (False, False, True, True)
    n_blocks: int = 4
    n_heads: int = 1
    n_groups: int = 32
    dropout_rate: float = 0.1
    label_dropout_rate: float = 0.0

    embedding_type: str = "positional"
    encoder_type : str = "standard"
    decoder_type : str = "standard"
    resample_filter: Union[Tuple[int, ...], List[int]] = (1, 1)
    learn_sigma: bool = False

    def setup(self):
        emb_channels = self.n_channels * 4
        noise_channels = self.n_channels * 1 # This can be changed
        init = create_initializer('xavier_uniform')
        init_zero = create_initializer('xavier_zero')
        init_attn = create_initializer('xavier_attn')
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
            skip_scale=jnp.sqrt(0.5),
            eps=1e-6,
            resample_filter=self.resample_filter,
            resample_proj=True,
            adaptive_scale=False
        )

        # Mapping
        self.map_noise = TimeEmbedding(noise_channels)
        self.map_label = nn.Dense(noise_channels, kernel_init=init) if self.label_dim else None
        self.map_augment = nn.Dense(noise_channels, use_bias=False, kernel_init=init) if self.augment_dim else None
        self.map_layer0 = nn.Dense(emb_channels, kernel_init=init)
        self.map_layer1 = nn.Dense(emb_channels, kernel_init=init)

        # Encoder
        enc_modules = {}
        skips = []
        cout = self.image_channels
        caux = self.image_channels
        for level, mult in enumerate(self.ch_mults):
            if level == 0:
                cin = cout
                cout = self.n_channels
                enc_modules[f'conv_{level}'] = CustomConv2d(in_channels=cin, out_channels=cout, kernel_channels=3, init_mode=init)
                skips.append(cout)
            else:
                enc_modules[f'down_{level}'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                skips.append(cout)
                if self.encoder_type == 'skip':
                    enc_modules[f'aux_down_{level}'] = CustomConv2d(in_channels=caux, out_channels=caux, kernel_channels=0, down=True, resample_filter=self.resample_filter)
                    enc_modules[f'aux_skip_{level}'] = CustomConv2d(in_channels=caux, out_channels=cout, kernel_channels=1, init_mode=init)
                if self.encoder_type == "residual":
                    enc_modules[f'aux_residual_{level}'] = CustomConv2d(in_channels=caux, out_channels=cout, kernel_channdls=3, down=True, resample_filter=self.resample_filter, fused_resample=True, init_mode=init)
                    caux=cout
            for idx in range(self.n_blocks):
                cin = cout
                cout = self.n_channels * mult
                attn = self.is_atten[level]
                enc_modules[f'{level}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
                skips.append(cout)

        dec_modules = {}
        for level, mult in reversed(list(enumerate(self.ch_mults))):
            if level == len(self.ch_mults) - 1:
                dec_modules[f"{level}_in0"] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                dec_modules[f"{level}_in1"] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                dec_modules[f"{level}_up"] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            
            for idx in range(self.n_blocks + 1):
                cin = cout + skips.pop()
                cout = self.n_channels * mult
                attn = (idx == self.n_blocks) and (self.is_atten[level])
                dec_modules[f"{level}_block{idx}"] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if self.decoder_type == 'skip' or level == 0:
                if self.decoder_type == "skip" and level < len(self.ch_mults) - 1:
                    dec_modules[f"{level}_aux_up"] = CustomConv2d(in_channels=self.image_channels, out_channels=self.image_channels, kernel_channels=0, up=True, resample_filter=self.resample_filter)
                dec_modules[f'{level}_aux_norm'] = nn.GroupNorm(epsilon=1e-6)
                dec_modules[f'{level}_aux_conv'] = CustomConv2d(in_channels=cout, out_channels=self.image_channels, kernel_channels=3, init_mode=init_zero)
        self.enc = enc_modules
        self.dec = dec_modules

    def __call__(self, x, noise_labels, train, augment_labels=None):
        emb = self.map_noise(noise_labels)
        # swap sin/cos
        emb_shape = emb.shape
        emb = emb.reshape(emb.shape[0], 2, -1)
        emb = jnp.flip(emb, axis=1)
        emb = emb.reshape(*emb_shape)
        # Add augment embedding if exists
        if augment_labels is not None:
            emb += self.map_augment(augment_labels)
        # augment_emb = jnp.where(augment_labels is None, jnp.zeros(emb.shape), self.map_augment(augment_labels))
        # emb += augment_emb

        # TODO: Add conditional stuffs in here
        emb = nn.silu(self.map_layer0(emb))
        emb = nn.silu(self.map_layer1(emb))

        # Encoder
        skips = []
        aux = x
        for key, block in self.enc.items():
            if 'aux_down' in key:
                aux = block(aux)
            elif 'aux_skip' in key:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in key:
                x = skips[-1] = aux = (x + block(aux)) / jnp.sqrt(2)
            else:
                x = block(x, emb, train) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder
        aux = None
        tmp = None
        for key, block in self.dec.items():
            if 'aux_up' in key:
                aux = block(aux)
            elif 'aux_norm' in key:
                tmp = block(x)
            elif 'aux_conv' in key:
                tmp = block(nn.silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[-1] != block.in_channels:
                    x = jnp.concatenate([x, skips.pop()], axis=-1)
                x = block(x, emb, train)
        return aux


class EDMPrecond(nn.Module):
    model_kwargs : dict

    image_channels: int
    label_dim: int = 0
    use_fp16: bool = False
    sigma_min : float = 0.0
    sigma_max : float = float('inf')
    sigma_data : float = 0.5
    model_type : str = "unetpp"
    
    @nn.compact
    def __call__(self, x, sigma, augment_labels, train):
        c_skip = (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)
        c_out = (sigma * self.sigma_data) / jnp.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1 / jnp.sqrt(self.sigma_data ** 2 + sigma ** 2)
        c_noise = jnp.log(sigma) / 4

        # Predict F_x. There is only UNetpp case for now. 
        # Should add more cases. (ex, DhariwalUNet (ADM)) 
        if self.model_type == "unetpp":
            net = UNetpp(**self.model_kwargs)
        elif self.model_type == "unet":
            net = UNet(**self.model_kwargs)
        
        F_x = net(c_in * x, c_noise.flatten(), train, augment_labels)
        D_x = c_skip * x + c_out * F_x
        return D_x