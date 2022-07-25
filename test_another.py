import torch
import torch.nn as nn

if __name__ == "__main__":
    mp = nn.MaxPool2d(3, 2, 1)
    a = torch.rand(4, 3, 512, 512)
    up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
    print(up(mp(a)).size())

