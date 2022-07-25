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


class CenterCombinePoolLayer(nn.Module):
    def __init__(self, in_channel):
        super(CenterCombinePoolLayer, self).__init__()
        self.max_pool = nn.MaxPool2d(3, 2, 1)
        self.avg_pool = nn.AvgPool2d(3, 2, 1)
        self.attn_ScaleConv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.attnConv1 = nn.Conv2d(in_channel * 2, in_channel, 1)
        self.attn_lastConv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
    def forward(self, x):
        attn_pre_1 = self.relu(self.attn_ScaleConv1(x))
        attn_max_1 = self.max_pool(attn_pre_1)
        attn_avg_1 = self.avg_pool(attn_pre_1)
        attn_sum_1 = self.relu(self.attnConv1(torch.cat((attn_max_1, attn_avg_1), dim=1)))
        attn_sum_1 = self.upsample(attn_sum_1)

        attn_sum_1_out = self.attn_lastConv1(attn_sum_1)
        attn = torch.sigmoid(attn_sum_1_out)

        out = x * attn + x
        return out

class CAResBlock(nn.Module):
    def __init__(self, channel = 64):
        super(CAResBlock, self).__init__()
        self.Conv1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.Conv2 = nn.Conv2d(channel,channel, 3, 1, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        res = x
        out1 = self.lrelu(self.Conv1(x))
        avg_mid = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_mid = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        mid = self.sigmoid(avg_mid + max_mid)
        out1 = out1 * mid
        out2 = self.lrelu(self.Conv2(out1))
        out = res + out2
        return out

# Time: 2022/7/21
class JointAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(JointAttention, self).__init__()
        self.depthConvBlock = CAResBlock(in_channel)
        self.rgbConvBlock = CenterCombinePoolLayer(in_channel)
        self.sigmoid = nn.Sigmoid()
    def forward(self, rgb, depth):
        res_depth = depth
        res_rgb = rgb
        emb_depth = self.sigmoid(self.depthConvBlock(depth))
        rgb_out = rgb * emb_depth + res_rgb

        emb_rgb = self.sigmoid(self.rgbConvBlock(rgb))
        depth_out = depth * emb_rgb + res_depth

        return rgb_out, depth_out