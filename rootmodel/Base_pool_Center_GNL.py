import torch
import torch.nn as nn
from rootmodel.model_utils import *
from rootmodel.SubNonLocalAttention import *

class CenterCombinePoolLayer(nn.Module):
    def __init__(self, in_channel):
        super(CenterCombinePoolLayer, self).__init__()
        self.max_pool = nn.MaxPool2d(1)
        self.avg_pool = nn.AvgPool2d(1)
        self.attn_ScaleConv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.attnConv1 = nn.Conv2d(in_channel * 2, in_channel, 1)
        self.attn_lastConv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        attn_pre_1 = self.lrelu(self.attn_ScaleConv1(x))
        attn_max_1 = self.max_pool(attn_pre_1)
        attn_avg_1 = self.avg_pool(attn_pre_1)
        attn_sum_1 = self.lrelu(self.attnConv1(torch.cat((attn_max_1, attn_avg_1), dim=1)))

        attn_sum_1_out = self.attn_lastConv1(attn_sum_1)
        attn = torch.sigmoid(attn_sum_1_out)

        out = x * attn + x
        return out

class CenterCombineAttention(nn.Module):
    def __init__(self, in_channel=64):
        super(CenterCombineAttention, self).__init__()
        # temporal attention (before fusion conv)
        self.max_pool = nn.MaxPool2d(1)
        self.avg_pool = nn.AvgPool2d(1)
        self.rgb_firstConv = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.depth_firstConv = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.fuse_Conv = nn.Conv2d(in_channel * 2, in_channel, 3, 1, 1)
        self.fuse_outConv = nn.Conv2d(in_channel, in_channel, 3, 1, 1)

        self.Up1_pre = PixUpBlock(512)
        self.CCP1 = CenterCombinePoolLayer(512)
        self.Up1_after = PixUpBlock(512)
        self.Up2_pre = PixUpBlock(128)
        self.CCP2 = CenterCombinePoolLayer(128)
        self.Up2_after = PixUpBlock(128)
        self.Up3_pre = PixUpBlock(32)
        self.CCP3 = CenterCombinePoolLayer(32)
        self.Up3_after = PixUpBlock(32)

        # self.Uplast = PixUpBlock(8)
        self.ConvLast = BasicConv2d(8, 8, 3)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
    def forward(self, rgb, depth):
        b, c, h, w = rgb.size()
        # temporal attention
        embedding_rgb = self.rgb_firstConv(rgb.clone())
        embedding_depth = self.depth_firstConv(depth.clone())
        corr = torch.sum(embedding_rgb * embedding_depth, 1).unsqueeze(1)
        fuse_fea = self.lrelu(self.fuse_Conv(torch.cat((embedding_rgb, embedding_depth), dim=1)))
        fuse_fea = corr * fuse_fea
        fuse_out = self.lrelu(self.fuse_outConv(fuse_fea))
        # print(fuse_out.size())
        # C=512, H=16, W=16

        Up1_pre = self.Up1_pre(fuse_out)
        Up1_after = self.Up1_after(self.CCP1(fuse_out))

        Up2_pre = self.Up2_pre(Up1_pre)
        Up2_after = self.Up2_after(self.CCP2(Up1_after+Up1_pre))

        Up3_pre = self.Up3_pre(Up2_pre)
        Up3_after = self.Up3_after(self.CCP3(Up2_after+Up2_pre))

        out = self.ConvLast(Up3_pre+Up3_after)
        # C=8, H=128, W=128
        # print(out.size())
        # print("asfdafas")
        return out

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

class FusionAttention(nn.Module):
    def __init__(self, in_channel = 64):
        super(FusionAttention, self).__init__()
        # temporal attention (before fusion conv)
        self.max_pool = nn.MaxPool2d(1)
        self.avg_pool = nn.AvgPool2d(1)
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

        attn_pre_1 = self.lrelu(self.attn_ScaleConv1(fuse_fea))
        attn_max_1 = self.max_pool(attn_pre_1)
        attn_avg_1 = self.avg_pool(attn_pre_1)
        attn_sum_1 = self.lrelu(self.attnConv1(torch.cat((attn_max_1, attn_avg_1), dim=1)))

        # attn_pre_2 = self.lrelu(self.attn_ScaleConv2(fuse_fea.view(b, -1, h//2, w//2)))
        # attn_max_2 = self.max_pool(attn_pre_2)
        # attn_avg_2 = self.avg_pool(attn_pre_2)
        # attn_sum_2 = self.lrelu(self.attnConv1(torch.cat((attn_max_2, attn_avg_2), dim=1)))

        attn_sum_1_out = self.attn_lastConv1(attn_sum_1)
        # attn_sum_2_up = self.attn_lastConv2(self.Up(attn_sum_2))
        attn = torch.sigmoid(attn_sum_1_out)

        out = fuse_out * attn + fuse_out
        return out

class PixUpBlock(nn.Module):
    def __init__(self, in_channel):
        super(PixUpBlock, self).__init__()
        self.up = nn.PixelShuffle(2)
        self.conv = BasicConv2d(in_channel//4, in_channel//4, 3)
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def forward(self, x):
        # print("pix before:",x.size())
        out = self.conv(self.up(x))
        # print("pix:",out.size())
        return out

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

    def forward(self, rgb_input, depth_input):
        rgb_1 = self.rgb_VGG_con1(rgb_input)
        depth_1 = self.dep_VGG_con1(depth_input)
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

        self.CCA = CenterCombineAttention(in_channel=512)
        # self.CenterUpConvLast = nn.Sequential(
        #     PixUpBlock(8),
        #     nn.Conv2d(2, 2, 3, 1, 1)
        # )

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

        self.depthUpblock2 = nn.Sequential(
            PixUpBlock(512),
            PixUpBlock(128)
        )

        self.depthUpblock3 = nn.Sequential(
            PixUpBlock(512),
            PixUpBlock(128),
            PixUpBlock(32),
            nn.Conv2d(8, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.fattn1 = FusionAttention(32)
        self.fattn2 = FusionAttention(32)
        self.fattn3 = FusionAttention(32)

        self.agg_block_1 = PixUpBlock(32)
        self.agg_block_2 = PixUpBlock(32)
        self.agg_block_3 = PixUpBlock(32)

        self.lastUp = nn.Sequential(
            PixUpBlock(8*4)
        )
        self.last_conv = BasicConv2d(8*4, 8*4, 3)
        self.GNLA = SubNonLocalAttention(2, 3, 8, 8, 32, 4)
        self.out_conv = nn.Conv2d(2*4, 1, 3, 1, 1)

        # self.Decoder = nn.Sequential()

    def forward(self, low_input, high_input):
        # VGG
        rgb_1, rgb_2, rgb_3, depth_1, depth_2, depth_3 = self.MSJCA(low_input, high_input)
        center = self.CCA(rgb_2, depth_2)
        # print("pre:")
        # print(rgb_1.size())
        # print(rgb_2.size())
        # print(rgb_3.size())
        # print(depth_1.size())
        # print(depth_2.size())
        # print(depth_3.size())
        # torch.Size([4, 256, 32, 32])
        # torch.Size([4, 512, 16, 16])
        # torch.Size([4, 512, 16, 16])
        # torch.Size([4, 256, 32, 32])
        # torch.Size([4, 512, 16, 16])
        # torch.Size([4, 512, 16, 16])
        rgb_1 = self.rgbUpblock1(rgb_1)
        rgb_2 = self.rgbUpblock2(rgb_2)
        rgb_3 = self.rgbUpblock3(rgb_3)

        depth_1 = self.depthUpblock1(depth_1)
        depth_2 = self.depthUpblock2(depth_2)
        depth_3 = self.depthUpblock3(depth_3)
        # print("then:")
        # print(rgb_1.size())
        # print(rgb_2.size())
        # print(rgb_3.size())
        # print(depth_1.size())
        # print(depth_2.size())
        # print(depth_3.size())
        # torch.Size([4, 32, 64, 64])
        # torch.Size([4, 32, 64, 64])
        # torch.Size([4, 32, 128, 128])
        # torch.Size([4, 32, 64, 64])
        # torch.Size([4, 32, 64, 64])
        # torch.Size([4, 32, 128, 128])

        fa_1 = self.fattn1(rgb_1, depth_1) # 128, 128, 64
        fa_2 = self.fattn2(rgb_2, depth_2) # 128, 128, 32
        fa_3 = self.fattn3(rgb_3, depth_3) # 64, 64 ,32
        # 全部 128， 128 ，32
        # print("after")
        # print(fa_1.size())
        # print(fa_2.size())
        # print(fa_3.size())
        # torch.Size([4, 32, 64, 64])
        # torch.Size([4, 32, 64, 64])
        # torch.Size([4, 32, 128, 128])
        fa_1_c = self.agg_block_1(fa_1)
        fa_2_c = self.agg_block_2(fa_2)+fa_1_c
        fa_3_c = self.agg_block_3(fa_3)+fa_2_c


        sum_c = torch.cat((fa_3_c, fa_2_c, fa_1_c, center), dim=1) # c=32*3
        # print("sum_c:", sum_c.size())
        last = self.last_conv(sum_c)
        last = self.GNLA(last, last)
        # print("last_conv:", last.size())
        last = self.lastUp(last)
        # print("last_up:", last.size())
        # print(last.size())
        # print(center.size())
        out = self.out_conv(last)
        # print("out:", out.size())
        return out

