import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

from einops import rearrange 

def Normalize(num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, epsilon=1e-6)

def nonlinearity(x):
    # swish
    return x * jax.nn.sigmoid(x)

class Upsample(nn.Module):
    """Upsample."""
    in_channels: int
    with_conv: bool

    def setup(self):
        if self.with_conv:
            self.conv = nn.Conv(self.in_channels,
                                kernel_size=(3, 3),
                                strides=1,
                                padding=1)

    def __call__(self, x):
        shape = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3])
        x = jax.image.resize(x, shape=shape, method="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Downsample."""
    in_channels: int
    with_conv: bool

    def setup(self):
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv(self.in_channels,
                                kernel_size=(3, 3),
                                strides=2,
                                padding=0)

    def __call__(self, x):
        if self.with_conv:
            pad_width = ((0, 0), (0, 1), (0, 1), (0, 0))
            x = jnp.pad(x, pad_width, mode="constant", constant_values=0)
            x = self.conv(x)
        else:
            x = jax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        return x


class ResnetBlock(nn.Module):
    """ResNet block."""

    in_channels: int
    out_channels: int
    dropout_rate: float
    temb_channels: int = 512
    use_conv_shortcut: bool = False

    def setup(self):

        self.norm1 = Normalize()
        self.conv1 = nn.Conv(self.out_channels,
                            kernel_size=(3, 3),
                            strides=1,
                            padding=1)
        if self.temb_channels > 0:
            self.temb_proj = nn.Dense(self.out_channels)

        self.norm2 = Normalize()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.conv2 = nn.Conv(self.out_channels,
                                kernel_size=(3, 3),
                                strides=1,
                                padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv(self.out_channels,
                                               kernel_size=(3, 3),
                                                strides=1,
                                                padding=1)
            else:
                self.nin_shortcut = nn.Conv(self.out_channels,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding=0)

    def __call__(self, x, temb, train):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h, deterministic=not train)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h
    

class AttnBlock(nn.Module):
    in_channels: int

    def setup(self):
        self.norm = Normalize(self.in_channels)
        self.q = nn.Conv(self.in_channels, kernel_size=(1, 1), strides=(1, 1), padding="VALID")
        self.k = nn.Conv(self.in_channels, kernel_size=(1, 1), strides=(1, 1), padding="VALID")
        self.v = nn.Conv(self.in_channels, kernel_size=(1, 1), strides=(1, 1), padding="VALID")
        self.proj_out = nn.Conv(self.in_channels, kernel_size=(1, 1), strides=(1, 1), padding="VALID")

    def __call__(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # To mimic the original implementation, we reshape the tensors to (b, h*w, c) and (b, c, h*w)
        # instead of (b, h, w, c) and (b, c, h, w)
        q = jnp.transpose(q, (0, 3, 1, 2))  # b, c, h, w
        k = jnp.transpose(k, (0, 3, 1, 2))
        v = jnp.transpose(v, (0, 3, 1, 2))

        # compute attention
        b, c, h, w = q.shape
        q = jnp.reshape(q, (b, c, h * w)) # b, hw, c
        q = jnp.transpose(q, (0, 2, 1))  # b, c, hw
        k = jnp.reshape(k, (b, c, h * w)) # b, c, hw 
        w_ = jnp.matmul(q, k)  # b, hw, hw    w[b,i,j] = sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (c ** (-0.5))
        w_ = jax.nn.softmax(w_, axis=2)

        # attend to values
        v = jnp.reshape(v, (b, c, h * w)) # b, hw, c 
        w_ = jnp.transpose(w_, (0, 2, 1))  # b, hw, hw (first hw of k, second of q)
        h_ = jnp.matmul(v, w_)  # b, c, hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = jnp.reshape(h_, (b, c, h, w)) # b, c, h, w

        # back to channels last
        h_ = jnp.transpose(h_, (0, 2, 3, 1))

        h_ = self.proj_out(h_)

        return x + h_

class LinearAttention(nn.Module):
    dim: int

    def setup(self):
        hidden_dim = self.dim
        self.to_qkv = nn.Conv(features=hidden_dim * 3, kernel_size=(1, 1), strides=(1, 1), padding="VALID", kernel_init=nn.initializers.lecun_normal(), bias=False)
        self.to_out = nn.Conv(features=self.dim, kernel_size=(1, 1), strides=(1, 1), padding="VALID", kernel_init=nn.initializers.lecun_normal())

    def __call__(self, x):
        # b, c, h, w = x.shape
        b, h, w, c = x.shape
        qkv = self.to_qkv(x)
        qkv = jnp.transpose(qkv, (0, 3, 1, 2)) # b, c, h, w
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=1, qkv=3)
        k = jax.nn.softmax(k, axis=-1)
        context = jnp.einsum('bhdn,bhen->bhde', k, v)
        out = jnp.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=1, h=h, w=w)
        out = jnp.transpose(out, (0, 2, 3, 1))

        return self.to_out(out)

def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        # return nn.Identity(in_channels)
        NotImplementedError("no attention is not implemented yet")
    else:
        return LinearAttention(in_channels)