import torch
import torch.nn as nn
from model_utils import *

class JointAttention(nn.Module):
    def __init__(self, in_channel, ratio=16):
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

    def foward(self, rgb, depth):
        rgb_fea = self.avg_convFC(self.avg_pool(rgb))
        depth_fea = self.max_convFC(self.max_pool(depth))
        sum_fea = rgb_fea + depth_fea
        sum_fea = self.sum_convFC(sum_fea)
        rgb_out = rgb * sum_fea + rgb
        depth_out = depth * sum_fea + depth
        return rgb_out, depth_out

class FusionAttention(nn.Module):
    def __init__(self, in_channel = 64):
        super(FusionAttention, self).__init__()
        # temporal attention (before fusion conv)
        self.rgb_firstConv = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.depth_firstConv = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.fuse_Conv = nn.Conv2d(in_channel*2, in_channel,3,1,1)
        self.fuse_outConv = nn.Conv2d(in_channel, in_channel, 3, 1, 1)

        self.attn_ScaleConv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.attnConv1 = nn.Conv2d(in_channel * 2, in_channel, 1)
        self.attn_ScaleConv2 = nn.Conv2d(in_channel*2, in_channel*2, 3, 1, 1)
        self.attnConv2 = nn.Conv2d(in_channel * 4, in_channel*2, 1)
        self.attn_lastConv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.attn_lastConv2 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.Up = nn.PixelShuffle(2)


        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
    def forward(self, rgb, depth):
        b, c, h, w = rgb.size()
        # temporal attention
        embedding_rgb = self.rgb_firstConv(rgb.clone())
        embedding_depth = self.depth_firstConv(depth.clone())
        corr = torch.sum(embedding_rgb*embedding_depth, 1).unsqueeze(1)
        fuse_fea = self.lrelu(self.fuse_Conv(torch.cat((embedding_rgb, embedding_depth), dim=1)))
        fuse_fea = corr * fuse_fea
        fuse_out = self.lrelu(self.fuse_outConv(fuse_fea))

        attn_pre_1 = self.lrelu(self.attn_preConv(fuse_fea))
        attn_max_1 = self.max_pool(attn_pre_1)
        attn_avg_1 = self.avg_pool(attn_pre_1)
        attn_sum_1 = self.lrelu(self.attnConv1(torch.cat((attn_max_1, attn_avg_1), dim=1)))

        attn_pre_2 = self.lrelu(self.attn_ScaleConv2(fuse_fea.view(b, -1, h//2, w//2)))
        attn_max_2 = self.max_pool(attn_pre_2)
        attn_avg_2 = self.avg_pool(attn_pre_2)
        attn_sum_2 = self.lrelu(self.attnConv1(torch.cat((attn_max_2, attn_avg_2), dim=1)))

        attn_sum_1_out = self.attn_lastConv1(attn_sum_1)
        attn_sum_2_up = self.attn_lastConv2(self.Up(attn_sum_2))
        attn = torch.sigmoid(attn_sum_1_out+attn_sum_2_up)

        out = fuse_out * attn + fuse_out
        return out

class PixUpBlock(nn.Module):
    def __init__(self, in_channel):
        super(PixUpBlock, self).__init__()
        self.up = nn.PixelShuffle(2)
        self.conv = BasicConv2d(in_channel//4, in_channel//4, 3)
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def foward(self, x):
        return self.conv(self.up(x))

class MSJCA(nn.Module):
    def __init__(self):
        super(MSJCA, self).__init__()
        self.rgb_VGG_con1 = VGG_conv1()
        self.rgb_VGG_con2 = VGG_conv2()
        self.rgb_VGG_con3 = VGG_conv3()
        self.rgb_VGG_con4 = VGG_conv4()
        self.rgb_VGG_con5 = VGG_conv5()

        self.dep_VGG_con1 = VGG_conv1()
        self.dep_VGG_con2 = VGG_conv2()
        self.dep_VGG_con3 = VGG_conv3()
        self.dep_VGG_con4 = VGG_conv4()
        self.dep_VGG_con5 = VGG_conv5()

        self.JA1 = JointAttention(in_channel=64, ratio=16)
        self.JA2 = JointAttention(in_channel=128, ratio=16)
        self.JA3 = JointAttention(in_channel=256, ratio=16)
        self.JA4 = JointAttention(in_channel=512, ratio=16)
        self.JA5 = JointAttention(in_channel=512, ratio=16)

    def foward(self, rgb_input, depth_input):
        rgb_1 = self.rgb_VGG_con1(rgb_input)
        depth_1 = self.dep_VGG_con2(depth_input)
        rgb_1, depth_1 = self.JA1(rgb_1, depth_1)

        rgb_2 = self.rgb_VGG_con2(rgb_1)
        depth_2 = self.dep_VGG_con2(depth_1)
        rgb_2, depth_2 = self.JA2(rgb_2, depth_2)

        rgb_3 = self.rgb_VGG_con3(rgb_2)
        depth_3 = self.dep_VGG_con3(depth_2)
        rgb_3, depth_3 = self.JA3(rgb_3, depth_3)

        rgb_4 = self.rgb_VGG_con4(rgb_3)
        depth_4 = self.dep_VGG_con4(depth_3)
        rgb_4, depth_4 = self.JA4(rgb_4, depth_4)

        rgb_5 = self.rgb_VGG_con5(rgb_4)
        depth_5 = self.dep_VGG_con5(depth_4)
        rgb_5, depth_5 = self.JA5(rgb_5, depth_5)

        ## remember 3,4,5
        return rgb_3, rgb_4, rgb_5, depth_3, depth_4, depth_5

# 可能的总体模型
class MSSOD(nn.Module):
    def __init__(self, rfb_out_channel=32):
        super(MSSOD, self).__init__()
        self.MSJCA = MSJCA()

        self.rgbUpblock1 = nn.Sequential(
            PixUpBlock(256),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.rgbUpblock2 = nn.Sequential(
            PixUpBlock(512),
            PixUpBlock(128)
        )

        self.rgbUpblock3 = nn.Sequential(
            PixUpBlock(512),
            PixUpBlock(128),
            PixUpBlock(32),
            nn.Conv2d(8, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.depthUpblock1 = nn.Sequential(
            PixUpBlock(256),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.depthpblock2 = nn.Sequential(
            PixUpBlock(512),
            PixUpBlock(128)
        )

        self.depthUpblock3 = nn.Sequential(
            PixUpBlock(512),
            PixUpBlock(128),
            nn.Conv2d(8, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.fattn1 = FusionAttention(64)
        self.fattn2 = FusionAttention(32)
        self.fattn3 = FusionAttention(32)

        self.agg_block_1 = PixUpBlock(64)
        self.agg_block_2 = PixUpBlock(32)
        self.agg_block_3 = PixUpBlock(32)

        self.lastUp = nn.Sequential(
            PixUpBlock(32*3),
            PixUpBlock(8*3)
        )
        self.last_conv = BasicConv2d(32*3, 32*3, 3)
        self.out_conv = nn.Conv2d(2*3, 1, 3)

        # self.Decoder = nn.Sequential()

    def forward(self, low_input, high_input):
        # VGG
        rgb_1, rgb_2, rgb_3, depth_1, depth_2, depth_3 = self.MSJCA(low_input, high_input)
        rgb_1 = self.rgbUpblock1(rgb_1)
        rgb_2 = self.rgbUpblock2(rgb_2)
        rgb_3 = self.rgbUpblock3(rgb_3)

        depth_1 = self.depthUpblock1(depth_1)
        depth_2 = self.depthUpblock2(depth_2)
        depth_3 = self.depthUpblock3(depth_3)


        fa_1 = self.fattn1(rgb_1, depth_1) # 128, 128, 64
        fa_2 = self.fattn2(rgb_2, depth_2) # 128, 128, 32
        fa_3 = self.fattn3(rgb_3, depth_3) # 64, 64 ,32
        # 全部 128， 128 ，32

        fa_1_c = self.agg_block_1(fa_1)
        fa_2_c = self.agg_block_2(fa_2)+fa_1_c
        fa_3_c = self.agg_block_3(fa_3)+fa_2_c


        sum_c = torch.cat((fa_3_c, fa_2_c, fa_1_c), dim=1) # c=32*3
        last = self.last_conv(sum_c)
        last = self.lastUp(last)
        out = self.out_conv(last)

        return out

