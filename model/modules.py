import jax
import jax.numpy as jnp
import flax.linen as nn

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
        x_skip = x
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
        res += x_skip

        return res


class UnetDown(nn.Module):
    out_channels: int
    has_atten: bool
    dropout_rate: float
    n_groups: int
    
    @nn.compact
    def __call__(self, x, t, train):
        x = ResidualBlock(self.out_channels, dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)
        if self.has_atten:
            x = AttentionBlock(self.out_channels, n_groups=self.n_groups)(x)
        return x

class UnetUp(nn.Module):
    out_channels: int
    has_atten: bool
    dropout_rate: float
    n_groups: int
    
    @nn.compact
    def __call__(self, x, t, train):
        x = ResidualBlock(self.out_channels, dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)
        if self.has_atten:
            x = AttentionBlock(self.out_channels, n_groups=self.n_groups)(x)
        return x

class UnetMiddle(nn.Module):
    n_channels: int
    dropout_rate: float
    n_groups: int

    @nn.compact
    def __call__(self, x, t, train):
        x = ResidualBlock(self.n_channels, dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)
        x = AttentionBlock(self.n_channels)(x)
        x = ResidualBlock(self.n_channels, dropout_rate=self.dropout_rate, n_groups=self.n_groups)(x, t, train)
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
        # B, H, W, C = x.shape
        # x = jnp.reshape(x, (B, H // 2, W // 2, C * 4))
        # x = nn.Conv(self.n_channels, (3, 3), (1, 1))(x)
        x = nn.Conv(self.n_channels, (3, 3), strides=2)(x)
        return x