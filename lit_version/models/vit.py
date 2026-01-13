import torch
import torch.nn as nn
from einops import rearrange  
import torch.nn.functional as F

import math

def get_2d_sincos_pos_embed(embed_dim, h, w, device):
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )

    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"

    omega = torch.arange(embed_dim // 4, device=device) / (embed_dim // 4)
    omega = 1. / (10000 ** omega)

    out_y = torch.einsum('hw,d->hwd', grid_y, omega)
    out_x = torch.einsum('hw,d->hwd', grid_x, omega)

    pos_emb = torch.cat(
        [torch.sin(out_y), torch.cos(out_y),
         torch.sin(out_x), torch.cos(out_x)],
        dim=-1
    )

    return pos_emb.reshape(h * w, embed_dim)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, stride=None):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        stride = stride or patch_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=stride)
        # Positional embeddings should be added


    def forward(self, x):
        patches = self.projection(x)
        h_patches, w_patches = patches.shape[2], patches.shape[3]
        patches = rearrange(patches, 'b e h w -> b (h w) e')
        # positional embeddings
        pos_emb = get_2d_sincos_pos_embed(patches.shape[-1], h_patches, w_patches, x.device)
        patches = patches + pos_emb.unsqueeze(0)
        return patches, h_patches, w_patches

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.GELU(),
            nn.Linear(emb_size * 4, emb_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, emb_size, num_layers, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(emb_size, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class UpsamplingDecoder(nn.Module):
    def __init__(self, emb_size, patch_size, out_channels, stride=None):
        """
        Upsamples the encoded representation back into an image.

        emb_size (int): Dimension of each patch
        patch_size (int): Size of each square patch
        out_channels (int): Number of output channels
        """
        super(UpsamplingDecoder, self).__init__()
        self.patch_size = patch_size
        stride = stride or patch_size
        self.projection = nn.ConvTranspose2d(
            emb_size, out_channels,
            kernel_size=patch_size, stride=stride
        )

    def forward(self, x, h_patches, w_patches, target_h=None, target_w=None):
        x = rearrange(x, 'b (h w) e -> b e h w', h=h_patches, w=w_patches)
        x = self.projection(x)

        if target_h is not None and target_w is not None:
            pad_h = target_h - x.shape[-2]
            pad_w = target_w - x.shape[-1]
            if pad_h > 0 or pad_w > 0:
               x = F.pad(x, (0, pad_w, 0, pad_h))
        return x


# Modèle complet combinant embedding, encodeur Transformer et décodeur
class DownscalingViT(nn.Module):
    def __init__(self, in_channels, emb_size, patch_size, num_layers, num_heads,
                 dropout=0.1, out_channels=1, output_shape=(70, 100), stride=None, forcings_dim=2):
        """
        Vision Transformer pour des tâches de descente d’échelle 2D.
        """
        super(DownscalingViT, self).__init__()

        # Dimensions de sortie ciblées
        self.output_shape = output_shape  # (height, width)

        # Module d'encodage des patchs
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, stride=stride)
        
        if forcings_dim > 0:
            self.fc_forcings = nn.Linear(forcings_dim, emb_size)
        # Encodeur Transformer
        self.transformer = TransformerEncoder(emb_size, num_layers, num_heads, dropout)

        # Décodeur final
        self.decoder = UpsamplingDecoder(emb_size, patch_size, out_channels, stride=stride)
        self.out_ch = out_channels

    def forward(self, x, forcings=None):
        # x : (batch_size, in_channels, height, width)
        # Redimensionne l’entrée à la taille cible par interpolation nearest neighbor
        x = nn.functional.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=False)

        # Découpe l’image en patchs encodés
        patches, h_patches, w_patches = self.patch_embedding(x)

        # Ajout des forcings saisonniers si fournis
        if forcings is not None:
            forcings_emb = self.fc_forcings(forcings)  # (batch_size, emb_size)
            patches = patches + forcings_emb.unsqueeze(1)  # Broadcasting

        # Passage dans l’encodeur Transformer
        encoded_patches = self.transformer(patches)

        # Reconstruction finale de l’image
        x = self.decoder(encoded_patches, h_patches, w_patches,
                 target_h=self.output_shape[0], target_w=self.output_shape[1])
        # if out_ch == 3, apply activation functions
        if self.out_ch == 3:
            # First channel: ocurrence (sigmoid)
            x1 = torch.sigmoid(x[:, 0:1, :, :])
            # Second channel: shape_parameter (softplus)
            x2 = F.softplus(x[:, 1:2, :, :])
            # Third channel: scale_parameter (softplus)
            x3 = F.softplus(x[:, 2:3, :, :])
            x = torch.cat([x1, x2, x3], dim=1)
        return x


if __name__ == "__main__":
    # Test the model with random input
    model = DownscalingViT(
        in_channels=4,
        emb_size=128,
        patch_size=(4, 6),
        num_layers=4,
        num_heads=4,
        dropout=0.0,
        out_channels=3,
        output_shape=(70, 100),
        stride=(2, 3)
    )
    x = torch.randn(2, 4, 35, 50)  # batch_size=2, in_channels=4, height=35, width=50
    out = model(x)
    print(out.shape)  # Expected output shape: (2, 3, 70, 100)