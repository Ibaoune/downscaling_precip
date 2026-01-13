import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def pad_to_multiple(x, multiple=16, mode="reflect"):
    _, _, h, w = x.shape
    pad_h = (math.ceil(h / multiple) * multiple) - h
    pad_w = (math.ceil(w / multiple) * multiple) - w

    pad = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    return F.pad(x, pad, mode=mode), pad

def unpad(x, pad):
    _, pad_w, _, pad_h = pad
    return x[..., : x.shape[-2] - pad_h, : x.shape[-1] - pad_w]


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, init_features=64, output_shape=(70, 100), dropout=0.1, forcings_dim=2):
        super().__init__()
        self.output_shape = output_shape
        self.out_ch = out_channels
        if forcings_dim > 0:
            in_channels += forcings_dim
        # Encoder
        self.enc1 = DoubleConv(in_channels, init_features, dropout=dropout)
        self.enc2 = DoubleConv(init_features, init_features * 2, dropout=dropout)
        self.enc3 = DoubleConv(init_features * 2, init_features * 4)
        self.enc4 = DoubleConv(init_features * 4, init_features * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(init_features * 8, init_features * 8)

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = DoubleConv(init_features * 16, init_features * 4)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = DoubleConv(init_features * 8, init_features * 2)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConv(init_features * 4, init_features, dropout=dropout)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DoubleConv(init_features * 2, init_features, dropout=dropout)

        # Output
        self.out_conv = nn.Conv2d(init_features, out_channels, kernel_size=1)

    def forward(self, x, forcings=None):
        # Encoder
        # interpolate to output size 
        x = nn.functional.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=False)
        # pad input to be multiple of 2^4=16
        x, pad = pad_to_multiple(x, multiple=16, mode="reflect")
        if forcings is not None: # (b, 1, forcings_dim)
            # expand forcings to match spatial dimensions
            b, _, h, w = x.shape
            forcings_expanded = forcings.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
            x = torch.cat([x, forcings_expanded], dim=1)
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.enc4(self.pool(c3))

        # Bottleneck
        b = self.bottleneck(self.pool(c4))

        # Decoder
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, c4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, c3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, c2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, c1], dim=1))
        out = self.out_conv(d1)
        if self.out_ch == 3:
            # First channel: ocurrence (sigmoid)
            x1 = torch.sigmoid(out[:, 0:1, :, :])
            # Second channel: shape_parameter (softplus)
            x2 = F.softplus(out[:, 1:2, :, :])
            # Third channel: scale_parameter (softplus)
            x3 = F.softplus(out[:, 2:3, :, :])
            out = torch.cat([x1, x2, x3], dim=1)
        # unpad to original size
        out = unpad(out, pad)
        return out

if __name__ == "__main__":
    model = UNet(in_channels=20, out_channels=1)
    x = torch.randn(2, 20, 35, 50)  # batch_size=2, in_channels=20, height=35, width=50
    out = model(x)
    print(out.shape)  # Expected output shape: (2, 1, 70, 100)