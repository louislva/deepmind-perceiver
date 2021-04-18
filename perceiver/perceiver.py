import torch
from torch import nn
import torch.nn.functional as F
from .positional_encoding import fourier_features


class PerceiverAttentionBlock(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super(PerceiverAttentionBlock, self).__init__()

        self.layer_norm_x = nn.LayerNorm([d_model])
        self.layer_norm_1 = nn.LayerNorm([d_model])
        self.attention = nn.MultiheadAttention(
            d_model,
            heads,
            dropout=0.0,
            bias=True,
            add_bias_kv=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(d_model, d_model)
        self.layer_norm_2 = nn.LayerNorm([d_model])
        self.linear2 = nn.Linear(d_model, d_model)
        self.linear3 = nn.Linear(d_model, d_model)

    def forward(self, x, z_input):
        x = self.layer_norm_x(x)
        z = self.layer_norm_1(z_input)
        z, _ = self.attention(z, x, x)

        z = self.dropout(z)
        z = self.linear1(z)

        z = self.layer_norm_2(z)
        z = self.linear2(z)
        z = F.gelu(z)
        z = self.dropout(z)
        z = self.linear3(z)

        return z + z_input


class PerceiverBlock(nn.Module):
    def __init__(self, d_model, latent_blocks, dropout, heads):
        super(PerceiverBlock, self).__init__()

        self.cross_attention = PerceiverAttentionBlock(
            d_model, heads=1, dropout=dropout)
        self.latent_attentions = nn.ModuleList([
            PerceiverAttentionBlock(d_model, heads=heads, dropout=dropout) for _ in range(latent_blocks)
        ])

    def forward(self, x, z):
        z = self.cross_attention(x, z)
        for latent_attention in self.latent_attentions:
            z = latent_attention(z, z)
        return z


class PerceiverBlockRepeater(nn.Module):
    def __init__(self, module, repeats=1):
        super(PerceiverBlockRepeater, self).__init__()

        self.repeats = repeats
        self.module = module

    def forward(self, x, z):
        for _ in range(self.repeats):
            z = self.module(x, z)
        return z


class Perceiver(nn.Module):
    def __init__(self, input_channels, input_shape, fourier_bands, latents=64, d_model=32, heads=8, latent_blocks=6, dropout=0.1, layers=8):
        super(Perceiver, self).__init__()
        self.fourier_features = fourier_features(
            shape=input_shape, bands=fourier_bands)

        self.init_latent = nn.Parameter(torch.rand((latents, d_model)))
        self.embedding = nn.Conv1d(
            input_channels + self.fourier_features.shape[0], d_model, 1)

        self.block1 = PerceiverBlockRepeater(
            PerceiverBlock(d_model, latent_blocks=latent_blocks,
                           heads=heads, dropout=dropout),
            repeats=1
        )  # 1
        self.block2 = PerceiverBlockRepeater(
            PerceiverBlock(d_model, latent_blocks=latent_blocks,
                           heads=heads, dropout=dropout),
            repeats=max(layers - 1, 0)
        )  # 2-8

    def forward(self, x):
        batch_size = x.shape[0]
        # Transform our X (input)
        # x.shape = (batch_size, channels, width, height)

        pos = self.fourier_features.unsqueeze(0).expand(
            (batch_size,) + self.fourier_features.shape)
        # pos.shape = (batch_size, self.fourier_bands * 2 * 2, width, height)

        x = torch.cat([x, pos], dim=1)
        # x.shape = (batch_size, channels + pos_channels, width, height)

        x = x.view((x.shape[0], x.shape[1], -1))
        # x.shape = (batch_size, channels, pixels)

        x = self.embedding(x)
        # x.shape = (batch_size, d_model, pixels)
        x = x.permute(2, 0, 1)
        # x.shape (pixels, batch_size, d_model)

        # Transform our Z (latent)
        # z.shape = (latents, d_model)
        z = self.init_latent.unsqueeze(1)
        # z.shape = (latents, 1, d_model)
        z = z.expand(-1, x.shape[1], -1)
        # z.shape = (latents, batch_size, d_model)

        z = self.block1(x, z)
        z = self.block2(x, z)

        return z


class PerceiverLogits(nn.Module):
    def __init__(self, input_channels, input_shape, output_features, fourier_bands=4, latents=64, d_model=32, heads=8, latent_blocks=6, dropout=0.1, layers=8):
        super(PerceiverLogits, self).__init__()

        self.perceiver = Perceiver(
            input_channels=input_channels,
            input_shape=input_shape,
            fourier_bands=fourier_bands,
            latents=latents,
            d_model=d_model,
            heads=heads,
            latent_blocks=latent_blocks,
            dropout=dropout,
            layers=layers
        )

        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, output_features)

    def forward(self, x):
        # Run the Perceiver
        z = self.perceiver(x)

        # Let data data inside of each latent
        z = self.linear1(z)
        # Then average every latent
        z = z.mean(dim=0)
        # Then extract logits
        z = self.linear2(z)

        return F.log_softmax(z, dim=-1)
