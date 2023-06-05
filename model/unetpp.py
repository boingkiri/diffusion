import jax.numpy as jnp
from flax import linen as nn

from typing import Tuple, Union, List, Any
from collections import OrderedDict

from model.modules import *

from model.unet import UNet

class Linear(nn.Module):
    in_features: int
    out_features: int
    use_bias: bool = True
    init_mode: Any = "kaiming_normal"
    init_weight: float = 1.0
    init_bias: float = 0.0
    def setup(self):
        init_function = create_initializer(self.init_mode) if type(self.init_mode) is str else self.init_mode
        weight_init_function = lambda key, shape, dtype: init_function(key, shape, dtype) * self.init_weight
        bias_init_function = lambda key, shape, dtype, fan_in, fan_out: init_function(key, shape, dtype, fan_in=fan_in, fan_out=fan_out) * self.init_bias
        self.weight = self.param("weight", 
                                weight_init_function, (self.in_features, self.out_features), jnp.float32)
        self.bias = self.param("bias", 
                                bias_init_function, (self.out_features,), jnp.float32,
                                self.in_features, self.out_features) if self.use_bias else None

    def __call__(self, x):
        x = x @ self.weight
        if self.use_bias:
            x = x + self.bias
        return x

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
        self.weight = self.param("weight", 
                   weight_init_function, 
                   (self.kernel_channels, self.kernel_channels, self.in_channels, self.out_channels),
                   jnp.float32) \
                    if self.kernel_channels else None
        self.bias = self.param("bias", 
                   bias_init_function, 
                   (self.out_channels, ), 
                   jnp.float32,
                   self.kernel_channels*self.kernel_channels*self.in_channels,
                   self.kernel_channels*self.kernel_channels*self.out_channels) \
                   if self.kernel_channels and self.use_bias else None
        f = jnp.asarray(self.resample_filter)
        f = jnp.expand_dims(jnp.outer(f, f), axis=(-1, -2)) / (jnp.sum(f) ** 2)
        self.resample_filter_outer = f if self.up or self.down else None
        self.dim_spec = ('NHWC', 'HWIO', 'NHWC')

    # @nn.compact
    def __call__(self, x):
        w = self.weight if self.weight is not None else None
        b = self.bias if self.bias is not None else None
        f = self.resample_filter_outer if self.resample_filter_outer is not None else None
        w_pad = w.shape[0] // 2 if w is not None else 0
        f_pad = (f.shape[0] - 1) // 2 if f is not None else 0

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
                padding = [[(f.shape[0] - 1) - f_pad] * 2] * 2
                x = jax.lax.conv_general_dilated(x, jnp.tile((f * 4), [1, 1, 1, self.in_channels]), window_strides=(1, 1),
                                            padding=padding, lhs_dilation=(2, 2), rhs_dilation=None, dimension_numbers=self.dim_spec,
                                            feature_group_count=self.in_channels)
            if self.down:
                padding = [[f_pad] * 2] * 2
                x = jax.lax.conv_general_dilated(x, jnp.tile(f, [1, 1, 1, self.in_channels]), window_strides=(2, 2), 
                                                 padding=padding, dimension_numbers=self.dim_spec, feature_group_count=self.in_channels)

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

    def setup(self):
        init_attn = create_initializer("xavier_attn")
        init_zero = create_initializer("xavier_zero")
        self.qkv = CustomConv2d(self.out_channels, self.out_channels * 3, kernel_channels=1, init_mode=init_attn)
        self.proj = CustomConv2d(self.out_channels, self.out_channels, kernel_channels=1, init_mode=init_zero)
        self.norm2 = nn.GroupNorm(epsilon=self.eps)

    @nn.compact
    def __call__(self, x):
        orig_x = x
        x = self.norm2(x)
        qkv = self.qkv(x)
        qkv = jnp.transpose(qkv, (0, 3, 1, 2))
        qkv = qkv.reshape(x.shape[0] * self.num_heads, x.shape[-1] // self.num_heads, 3, -1)
        q, k, v = jnp.split(qkv, 3, axis=2)
        k = k / jnp.sqrt(k.shape[1])
        
        w = jnp.einsum('bcnq,bcnk->bnqk', q, k)
        w = nn.softmax(w, axis=-1)

        a = jnp.einsum('bnqk,bcnk->bcnq', w, v)
        a = a.reshape(x.shape[0], x.shape[3], x.shape[1], x.shape[2])
        a = jnp.transpose(a, (0, 2, 3, 1))
        
        x = self.proj(a) + orig_x
        # orig_x = x
        # x = self.norm2(x)
        # qkv = self.qkv(x)
        # qkv = qkv.reshape(x.shape[0] * self.num_heads, -1, 3, x.shape[-1] // self.num_heads)
        # q, k, v = jnp.split(qkv, 3, axis=2)
        # k = k / jnp.sqrt(k.shape[-1])
        
        # w = jnp.einsum('bqnc,bknc->bqkn', q, k)
        # w = nn.softmax(w, axis=2)

        # a = jnp.einsum('bqkn,bknc->bqnc', w, v)
        # a = a.reshape(*x.shape)
        # x = self.proj(a) + orig_x
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
        self.affine = Linear(self.emb_channels, self.out_channels * (2 if self.adaptive_scale else 1), init_mode=init)
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
                init_mode=init)
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

class PositionalEmbedding(nn.Module):
    num_channels: int
    max_positions: int = 10000
    endpoint: bool = False

    def __call__(self, x):
        freqs = jnp.arange(start=0, stop=self.num_channels // 2)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = jnp.outer(x, freqs.astype(x.dtype))
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=1)
        return x

class FourierEmbedding(nn.Module):
    num_channels: int
    scale: float = 16
    def setup(self):
        # key = self.make_rng('params')
        # randn = jax.random.normal(key, (self.num_channels // 2,))
        # self.freqs = randn * self.scale
        self.freqs = self.param('freqs', jax.random.normal, (self.num_channels // 2,), jnp.float32)
    
    def __call__(self, x):
        # x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        freqs = jax.lax.stop_gradient(self.freqs)
        x = jnp.outer(x, (2 * jnp.pi * freqs))
        x = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=1)
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

    t_emb_output: bool = False

    def setup(self):
        emb_channels = self.n_channels * 4
        # noise_channels = self.n_channels * 1 # This can be changed
        noise_channels = self.n_channels * (1 if len(self.resample_filter) == 2 else 2)
        init = create_initializer('xavier_uniform')
        init_zero = create_initializer('xavier_zero')
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
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if self.embedding_type == "positional" else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(self.label_dim, noise_channels, init_mode=init) if self.label_dim else None 
        self.map_augment = Linear(self.augment_dim, noise_channels, use_bias=False, init_mode=init) if self.augment_dim else None
        self.map_layer0 = Linear(noise_channels, emb_channels, init_mode=init)
        self.map_layer1 = Linear(emb_channels, emb_channels, init_mode=init)

        # Encoder
        enc_modules = {}
        skips = []
        cout = self.image_channels
        caux = self.image_channels

        # TMP: for CIFAR10
        img_res = 32

        for level, mult in enumerate(self.ch_mults):
            res = img_res >> level
            if level == 0:
                cin = cout
                cout = self.n_channels
                enc_modules[f'{res}x{res}_conv'] = CustomConv2d(in_channels=cin, out_channels=cout, kernel_channels=3, init_mode=init)
                skips.append(cout)
            else:
                enc_modules[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                skips.append(cout)
                if self.encoder_type == 'skip':
                    enc_modules[f'{res}x{res}_aux_down'] = CustomConv2d(in_channels=caux, out_channels=caux, kernel_channels=0, down=True, resample_filter=self.resample_filter)
                    enc_modules[f'{res}x{res}_aux_skip'] = CustomConv2d(in_channels=caux, out_channels=cout, kernel_channels=1, init_mode=init)
                if self.encoder_type == "residual":
                    enc_modules[f'{res}x{res}_aux_residual'] = CustomConv2d(in_channels=caux, out_channels=cout, kernel_channels=3, down=True, resample_filter=self.resample_filter, fused_resample=True, init_mode=init)
                    caux=cout
            for idx in range(self.n_blocks):
                cin = cout
                cout = self.n_channels * mult
                attn = self.is_atten[level]
                enc_modules[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
                skips.append(cout)

        dec_modules = {}
        for level, mult in reversed(list(enumerate(self.ch_mults))):
            res = img_res >> level
            if level == len(self.ch_mults) - 1:
                dec_modules[f"{res}x{res}_in0"] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                dec_modules[f"{res}x{res}_in1"] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                dec_modules[f"{res}x{res}_up"] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)

            for idx in range(self.n_blocks + 1):
                cin = cout + skips.pop()
                cout = self.n_channels * mult
                attn = (idx == self.n_blocks) and (self.is_atten[level])
                dec_modules[f"{res}x{res}_block{idx}"] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if self.decoder_type == 'skip' or level == 0:
                if self.decoder_type == "skip" and level < len(self.ch_mults) - 1:
                    dec_modules[f"{res}x{res}_aux_up"] = CustomConv2d(in_channels=self.image_channels, out_channels=self.image_channels, kernel_channels=0, up=True, resample_filter=self.resample_filter)
                dec_modules[f'{res}x{res}_aux_norm'] = nn.GroupNorm(epsilon=1e-6)
                dec_modules[f'{res}x{res}_aux_conv'] = CustomConv2d(in_channels=cout, out_channels=self.image_channels, kernel_channels=3, init_mode=init_zero)
        self.enc = enc_modules
        self.dec = dec_modules

    def __call__(self, x, noise_labels, train, augment_labels=None):
        emb = self.map_noise(noise_labels)

        # Swap sin/cos
        emb_shape = emb.shape
        emb = emb.reshape(emb.shape[0], 2, -1)
        emb = jnp.flip(emb, axis=1)
        emb = emb.reshape(*emb_shape)

        # Add augment embedding if exists
        if augment_labels is not None:
            emb += self.map_augment(augment_labels)

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
        last_xemb = None
        for key, block in self.dec.items():
            if 'aux_up' in key:
                aux = block(aux)
            elif 'aux_norm' in key:
                last_xemb = x
                tmp = block(x)
            elif 'aux_conv' in key:
                tmp = block(nn.silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[-1] != block.in_channels:
                    x = jnp.concatenate([x, skips.pop()], axis=-1)
                x = block(x, emb, train)
        
        if self.t_emb_output:
            return aux, emb, last_xemb
        else:
            return aux

class TimeEmbedDependentHead(nn.Module):
    image_channels: int = 3
    n_channels: int = 128
    last_ch_mult: int = 2
    n_blocks: int = 4
    n_heads: int = 1
    dropout_rate: float = 0.1
    resample_filter: Union[Tuple[int, ...], List[int]] = (1, 1)
    
    def setup(self):
        init = create_initializer("xavier_uniform")
        emb_channels = self.n_channels * 4
        
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
        # self.normalize1 = nn.GroupNorm(num_groups=self.image_channels)
        self.normalize1 = nn.GroupNorm()
        head_channels = self.n_channels * self.last_ch_mult
        self.conv1 = CustomConv2d(in_channels=head_channels + self.image_channels * 2, 
                                  out_channels=head_channels, 
                                  kernel_channels=3,
                                  init_mode=init)
        
        blocks = []
        for _ in range(self.n_blocks):
            blocks.append(UNetBlock(in_channels=head_channels, out_channels=head_channels, **block_kwargs))
        
        self.blocks = blocks
        self.normalize2 = nn.GroupNorm()
        self.conv2 = CustomConv2d(in_channels=head_channels,
                                    out_channels=self.image_channels,
                                    kernel_channels=3,
                                    init_mode=init)

    def __call__(self, x, x_pred, x_emb, t_emb, train):
        # x = self.conv1(nn.silu(self.normalize1(x)))
        x_emb_norm = self.normalize1(x_emb)
        x_emb = self.conv1(nn.silu(jnp.concatenate([x, x_pred, x_emb_norm], axis=-1)))
        for block in self.blocks:
            x_emb = block(x_emb, t_emb, train)
        x = self.conv2(nn.silu(self.normalize2(x_emb)))
        return x

        
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

class CMPrecond(nn.Module):
    model_kwargs : dict

    image_channels: int
    label_dim: int = 0
    use_fp16: bool = False
    sigma_min : float = 0.0 # 0.002
    sigma_max : float = float('inf') # 80
    sigma_data : float = 0.5
    model_type : str = "unetpp"
    
    @nn.compact
    def __call__(self, x, sigma, augment_labels, train):
        c_skip = (self.sigma_data ** 2) / ((sigma - self.sigma_min) ** 2 + self.sigma_data ** 2)
        c_out = ((sigma - self.sigma_min) * self.sigma_data) / jnp.sqrt(sigma ** 2 + self.sigma_data ** 2)
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

class CMDMPrecond(nn.Module):
    model_kwargs : dict
    image_channels: int

    label_dim: int = 0
    use_fp16: bool = False
    sigma_min : float = 0.0 # 0.002
    sigma_max : float = float('inf') # 80
    sigma_data : float = 0.5
    model_type : str = "unetpp"

    t_emb_output: bool = False

    def setup(self):
        if self.model_type == "unetpp":
            self.UNetpp_0 = UNetpp(**self.model_kwargs, t_emb_output=self.t_emb_output)
        elif self.model_type == "unet":
            self.UNetpp_0 = UNet(**self.model_kwargs)
        
        self.TimeEmbedDependentHead_0 = TimeEmbedDependentHead( # For consistency
            image_channels=self.image_channels, 
            n_channels=self.model_kwargs['n_channels'],
            n_heads=self.model_kwargs['n_heads'],
            dropout_rate=self.model_kwargs['dropout_rate'],
            resample_filter=self.model_kwargs['resample_filter'],)
        self.TimeEmbedDependentHead_1 = TimeEmbedDependentHead( # For diffusion
            image_channels=self.image_channels, 
            n_channels=self.model_kwargs['n_channels'],
            n_heads=self.model_kwargs['n_heads'],
            dropout_rate=self.model_kwargs['dropout_rate'],
            resample_filter=self.model_kwargs['resample_filter'],)
    
    # @nn.compact
    def __call__(self, x, sigma, augment_labels, train):
        c_skip = (self.sigma_data ** 2) / ((sigma - self.sigma_min) ** 2 + self.sigma_data ** 2)
        c_out = ((sigma - self.sigma_min) * self.sigma_data) / jnp.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1 / jnp.sqrt(self.sigma_data ** 2 + sigma ** 2)
        c_noise = jnp.log(sigma) / 4

        # Predict F_x. There is only UNetpp case for now. 
        # Should add more cases. (ex, DhariwalUNet (ADM)) 
        # if self.model_type == "unetpp":
        #     net = UNetpp(**self.model_kwargs, t_emb_output=self.t_emb_output)
        # elif self.model_type == "unet":
        #     net = UNet(**self.model_kwargs)

        init_zero = create_initializer('xavier_zero')
        
        # D_x = c_skip * x + c_out * F_x
        # return D_x
        # consistency_head = TimeEmbedDependentHead(
        #     image_channels=self.image_channels, 
        #     n_channels=self.model_kwargs['n_channels'],
        #     last_ch_mult=self.model_kwargs['ch_mults'][0],
        #     n_heads=self.model_kwargs['n_heads'],
        #     dropout_rate=self.model_kwargs['dropout_rate'],
        #     resample_filter=self.model_kwargs['resample_filter'],)
        # diffusion_head = TimeEmbedDependentHead(
        #     image_channels=self.image_channels, 
        #     n_channels=self.model_kwargs['n_channels'],
        #     last_ch_mult=self.model_kwargs['ch_mults'][0],
        #     n_heads=self.model_kwargs['n_heads'],
        #     dropout_rate=self.model_kwargs['dropout_rate'],
        #     resample_filter=self.model_kwargs['resample_filter'],)
        if not self.t_emb_output:
            F_x = self.UNetpp_0(c_in * x, c_noise.flatten(), train, augment_labels)
            t_emb = jnp.zeros((x.shape[0], self.model_kwargs['n_channels'] * 4))
            consistency_result = c_skip * x + c_out * self.TimeEmbedDependentHead_0(c_in * x, F_x, t_emb, train)
            diffusion_result = c_skip * x + c_out * self.TimeEmbedDependentHead_1(c_in * x, F_x, t_emb, train)
        else:
            F_x, t_emb, last_x_emb = self.UNetpp_0(c_in * x, c_noise.flatten(), train, augment_labels)
            consistency_result = c_skip * x + c_out * self.TimeEmbedDependentHead_0(c_in * x, F_x, last_x_emb, t_emb, train)
            diffusion_result = c_skip * x + c_out * self.TimeEmbedDependentHead_1(c_in * x, F_x, last_x_emb, t_emb, train)

        return F_x, consistency_result, diffusion_result
    
    def diffusion_output(self, x, sigma, augment_labels, train):
        c_skip = (self.sigma_data ** 2) / ((sigma - self.sigma_min) ** 2 + self.sigma_data ** 2)
        c_out = ((sigma - self.sigma_min) * self.sigma_data) / jnp.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1 / jnp.sqrt(self.sigma_data ** 2 + sigma ** 2)
        c_noise = jnp.log(sigma) / 4

        if not self.t_emb_output:
            F_x = self.UNetpp_0(c_in * x, c_noise.flatten(), train, augment_labels)
            t_emb = jnp.zeros((x.shape[0], self.model_kwargs['n_channels'] * 4))
            diffusion_result = c_skip * x + c_out * self.TimeEmbedDependentHead_1(F_x, t_emb, train)
        else:
            F_x, t_emb, last_x_emb = self.UNetpp_0(c_in * x, c_noise.flatten(), train, augment_labels)
            diffusion_result = c_skip * x + c_out * self.TimeEmbedDependentHead_1(c_in * x, F_x, last_x_emb, t_emb, train)

        return diffusion_result

    def consistency_output(self, x, sigma, augment_labels, train):
        c_skip = (self.sigma_data ** 2) / ((sigma - self.sigma_min) ** 2 + self.sigma_data ** 2)
        c_out = ((sigma - self.sigma_min) * self.sigma_data) / jnp.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1 / jnp.sqrt(self.sigma_data ** 2 + sigma ** 2)
        c_noise = jnp.log(sigma) / 4

        if not self.t_emb_output:
            F_x = self.UNetpp_0(c_in * x, c_noise.flatten(), train, augment_labels)
            t_emb = jnp.zeros((x.shape[0], self.model_kwargs['n_channels'] * 4))
            consistency_result = c_skip * x + c_out * self.TimeEmbedDependentHead_0(F_x, t_emb, train)
        else:
            F_x, t_emb, last_x_emb = self.UNetpp_0(c_in * x, c_noise.flatten(), train, augment_labels)
            consistency_result = c_skip * x + c_out * self.TimeEmbedDependentHead_0(c_in * x, F_x, last_x_emb, t_emb, train)

        return consistency_result