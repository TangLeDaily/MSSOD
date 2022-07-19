import torch
import torch.nn as nn

# Time: 2022/7/18

class JointAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(JointAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_convFC = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // ratio, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // ratio, in_channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_convFC = nn.Sequential(nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False),
                                nn.Sigmoid())
        #
        self.sum_convFC = nn.Sequential(nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False),
                                nn.Sigmoid())

    def forward(self, rgb, depth):
        rgb_fea = self.avg_convFC(self.avg_pool(rgb))
        depth_fea = self.max_convFC(self.max_pool(depth))
        sum_fea = rgb_fea + depth_fea
        sum_fea = self.sum_convFC(sum_fea)
        rgb_out = rgb * sum_fea + rgb
        depth_out = depth * sum_fea + depth
        return rgb_out, depth_out


# Time: 2022/7/19

class JointAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(JointAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_convFC = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // ratio, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // ratio, in_channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_convFC = nn.Sequential(nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False),
                                nn.Sigmoid())
        #
        self.sum_convFC = nn.Sequential(nn.Conv2d(in_channel, in_channel // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channel // ratio, in_channel, 1, bias=False),
                                nn.Sigmoid())
        self.Convrgb = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.Convdepth = nn.Conv2d(in_channel, in_channel, 3, 1, 1)

        self.outConvrgb = nn.Conv2d(in_channel*3, in_channel, 3, 1, 1)
        self.outConvdepth = nn.Conv2d(in_channel*3, in_channel, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, rgb, depth):
        rgb_fea = self.avg_convFC(self.avg_pool(rgb))
        depth_fea = self.max_convFC(self.max_pool(depth))
        sum_fea = rgb_fea + depth_fea
        sum_fea = self.sum_convFC(sum_fea)

        rgb_out1 = rgb * sum_fea + rgb
        depth_out1 = depth * sum_fea + depth

        rgb_out2 = self.Convrgb(rgb)*self.sigmoid(depth_out1)
        depth_out2 = self.Convdepth(depth)*self.sigmoid(rgb_out1)

        rgb_out = self.lrelu(self.outConvrgb(torch.cat([rgb, rgb_out1, rgb_out2], dim=1)))+rgb
        depth_out = self.lrelu(self.outConvrgb(torch.cat([depth, depth_out1, depth_out2], dim=1))) + depth

        return rgb_out, depth_out