import flax.linen as nn

from typing import Union, Tuple, List 
from modules import UnetUp, UnetMiddle, UnetDown, Downsample, Upsample


class Encoder(nn.Module):
    image_channels: int = 3
    n_channels: int = 128
    ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 4)
    is_atten: Union[Tuple[bool, ...], List[bool]] = (False, False, False)
    n_blocks: int = 2
    dropout_rate: float = 0.0
    n_heads: int = 1
    
    @nn.compact
    def __call__(self, x, train):
        t = None

        n_resolution = len(self.ch_mults)
        for i in range(n_resolution):
            out_channels = self.n_channels * self.ch_mults[i]
            for _ in range(self.n_blocks):
                x = UnetDown(out_channels, self.is_atten[i], dropout_rate=self.dropout_rate)(x, t, train)
            if i < n_resolution - 1:
                out_channels = self.n_channels * self.ch_mults[i+1]
                x = Downsample(out_channels)(x)
        
        x = UnetMiddle(out_channels, dropout_rate=self.dropout_rate)(x, t, train)

        x = nn.GroupNorm(8)(x)
        x = nn.swish(x)
        x = nn.Conv(self.n_channels * self.ch_mults[-1], (3, 3))(x)

        return x

class Decoder(nn.Module):
    image_channels: int = 3
    n_channels: int = 128
    ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 4)
    is_atten: Union[Tuple[bool, ...], List[bool]] = (False, False, False)
    n_blocks: int = 2
    dropout_rate: float = 0.0
    n_heads: int = 1
    
    @nn.compact
    def __call__(self, x, train):
        t = None
        n_resolution = len(self.ch_mults)

        x = nn.Conv(self.n_channels, (3, 3))(x)
        x = UnetMiddle(out_channels, dropout_rate=self.dropout_rate)(x, t, train)
        
        for i in reversed(range(n_resolution)):
            out_channels = self.n_channels * self.ch_mults[i]
            for _ in range(self.n_blocks):
                x = UnetUp(out_channels, self.is_atten[i], dropout_rate=self.dropout_rate)(x, t, train)
            if i > 0:
                out_channels = self.n_channels * self.ch_mults[i - 1]
                x = Upsample(out_channels)(x)

        x = nn.GroupNorm(8)(x)
        x = nn.swish(x)
        x = nn.Conv(self.image_channels, (3, 3))(x)

        return x

class AutoEncoder(nn.Module):
    def __init__(
        self, 
        image_channels: int = 3,
        n_channels: int = 128,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 4),
        is_atten: Union[Tuple[bool, ...], List[bool]] = (False, False, False),
        n_blocks: int = 2,
        dropout_rate: float = 0.1,
        n_heads: int = 1
    ):
        params = [
            image_channels,
            n_channels,
            ch_mults,
            is_atten,
            n_blocks,
            dropout_rate,
            n_heads
        ]
        self.encoder_model = Encoder(*params)
        self.decoder_model = Decoder(*params)

    def __call__(self, x, train):
        z = self.encoder(x, train)
        x_rec = self.decoder(z, train)
        return x_rec

    def encoder(self, x, train):
        return self.encoder_model(x, train)

    def decoder(self, z, train):
        return self.decoder_model(z, train)