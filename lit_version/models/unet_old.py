import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, init_features=64, output_shape=(70, 100)):
        super().__init__()

        # Encoder
        self.enc_conv1 = self.conv_block(in_channels, 64, kernel_size=2, stride=1, padding=1)
        self.enc_conv2 = self.conv_block(64, 128, kernel_size=2, stride=1, padding=1)
        self.enc_conv3 = self.conv_block(128, 256, kernel_size=2, stride=1, padding=1)
        self.enc_conv4 = self.conv_block(256, 512, kernel_size=2, stride=1, padding=1)
        self.enc_conv5 = self.conv_block(512, 1024, kernel_size=2, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2)
        # Decoder
        self.dec_conv4 = self.conv_block(1024+512 , 512, kernel_size=2, padding=1)  # Modified to match the input c hannels
        self.dec_conv3 = self.conv_block(512+256, 256, kernel_size=2, padding=1)
        self.dec_conv2 = self.conv_block(256+128, 128, kernel_size=2, padding=1)
        self.dec_conv1 = self.conv_block(128+64, 64, kernel_size=2, padding=1)

        self.upconv4 = self.conv_trans(1024, 1024, kernel_size=2, stride=2, out=(1, 1))
        self.upconv3 = self.conv_trans(512, 512, kernel_size=2, stride=2, out=(1, 1))
        self.upconv2 = self.conv_trans(256, 256, kernel_size=2, stride=2, out=(1, 1))
        self.upconv1 = self.conv_trans(128,128, kernel_size=2, stride=2, out=(1, 1))

        self.ad = nn.Conv2d(64, 64, kernel_size=(5,2), stride=1, padding=(1, 1))
        self.upconv05 =nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2,padding=0, output_padding=(1, 1))
        self.upconv03 =nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1,padding=0, output_padding=(0, 0))
        self.upconv0 =nn.ConvTranspose2d(64, out_channels, kernel_size=5, stride=1,padding=0, output_padding=(0, 0))

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        _, _, target_height, target_width = target_size
        delta_h = (layer_height - target_height) // 2
        delta_w = (layer_width - target_width) // 2
        return layer[:, :, delta_h:delta_h + target_height, delta_w:delta_w + target_width]

        
    def conv_block(self, in_channels, out_channels, kernel_size=2,stride=1, padding=1):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size=kernel_size, 
                      stride=stride, 
                      padding=padding
                      ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        return block
        
    def conv_trans(self, in_channels, out_channels, kernel_size=2, stride=2,padding=0, out=(1, 1)):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 
                               kernel_size=kernel_size, 
                               stride=stride,
                               padding=padding, 
                               output_padding=out
                               ),
            nn.ReLU(),
        )
        return block

        

    def forward(self, x):
        # Encoder
        c1 = self.enc_conv1(x)
        p1 = self.pool(c1)
    
        c2 = self.enc_conv2(p1)
        p2 = self.pool(c2)
    
        c3 = self.enc_conv3(p2)
        p3 = self.pool(c3)
    
        c4 = self.enc_conv4(p3)
        p4 = self.pool(c4)
    
        c5 = self.enc_conv5(p4)
    
        # Decoder
        u4 = self.upconv4(c5)
        u4 = self.center_crop(u4, c4.size())
        u4 = torch.cat([u4, c4], dim=1)
        c6 = self.dec_conv4(u4)
    
        u3 = self.upconv3(c6)
        u3 = self.center_crop(u3, c3.size())
        u3 = torch.cat([u3, c3], dim=1)
        c7 = self.dec_conv3(u3)
    
        u2 = self.upconv2(c7)
        u2 = self.center_crop(u2, c2.size())
        u2 = torch.cat([u2, c2], dim=1)
        c8 = self.dec_conv2(u2)
    
        u1 = self.upconv1(c8)
        u1 = self.center_crop(u1, c1.size())
        u1 = torch.cat([u1, c1], dim=1)
        c9 = self.dec_conv1(u1)

        # Final upsampling layers
        c9 = F.relu(self.ad(c9))
        c9 = F.relu(self.upconv05(c9))
        c9 = F.relu(self.upconv03(c9))
        out = self.upconv0(c9)

        return out

if __name__ == "__main__":
    model = UNet()
    x = torch.randn(2, 20, 35, 50)  # batch_size=2, in_channels=20, height=35, width=50
    out = model(x)
    print(out.shape)  # Expected output shape: (2, 1, 70, 100)