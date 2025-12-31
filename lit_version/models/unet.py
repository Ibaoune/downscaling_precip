import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, init_features=64, output_shape=(70, 100)):
        super(UNet, self).__init__()
        self.output_shape = output_shape
        features = init_features
        
        # Encoder
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._block(features * 2, features * 4)
        
        # Decoder
                
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)
        
        # Output
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)
        self.out_ch = out_channels
    
    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding="same"),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding="same"),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = nn.functional.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=False)
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        # Bottleneck
        bottleneck = self.bottleneck(self.pool2(enc2))
        # Decoder with skip connections
        dec2 = self.upconv2(bottleneck)
        # add padding if needed to match enc2 size
        if dec2.size() != enc2.size():
            dec2 = nn.functional.pad(dec2, (0, enc2.size(3) - dec2.size(3), 0, enc2.size(2) - dec2.size(2)))
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        # add padding if needed to match enc1 size
        if dec1.size() != enc1.size():
            dec1 = nn.functional.pad(dec1, (0, enc1.size(3) - dec1.size(3), 0, enc1.size(2) - dec1.size(2)))
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        x = self.conv(dec1)
        
        if self.out_ch == 3:
            # First channel: ocurrence (sigmoid)
            x1 = torch.sigmoid(x[:, 0:1, :, :])
            # Second channel: shape_parameter (softplus)
            x2 = F.softplus(x[:, 1:2, :, :])
            # Third channel: scale_parameter (softplus)
            x3 = F.softplus(x[:, 2:3, :, :])
            x = torch.cat([x1, x2, x3], dim=1)
        return x
    
