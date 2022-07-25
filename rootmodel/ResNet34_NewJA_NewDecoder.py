import torch
import torch.nn as nn
from rootmodel.model_utils import *

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

class CenterCombineAttention(nn.Module):
    def __init__(self, in_channel=64):
        super(CenterCombineAttention, self).__init__()
        # temporal attention (before fusion conv)

        self.rgb_firstConv = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.depth_firstConv = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.fuse_Conv = nn.Conv2d(in_channel * 2, in_channel, 3, 1, 1)
        self.fuse_outConv = nn.Conv2d(in_channel, in_channel, 3, 1, 1)

        self.Up1_pre = PixUpBlock(in_channel)
        self.CCP1 = CenterCombinePoolLayer(in_channel)
        self.Up1_after = PixUpBlock(in_channel)
        self.Up2_pre = PixUpBlock(in_channel//4)
        self.CCP2 = CenterCombinePoolLayer(in_channel//4)
        self.Up2_after = PixUpBlock(in_channel//4)
        self.Up3_pre = PixUpBlock(in_channel//16)
        self.CCP3 = CenterCombinePoolLayer(in_channel//16)
        self.Up3_after = PixUpBlock(in_channel//16)

        self.ConvLast = BasicBlock(in_channel//64, in_channel//64)

        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
    def forward(self, rgb, depth):
        b, c, h, w = rgb.size()
        # temporal attention
        embedding_rgb = self.rgb_firstConv(rgb.clone())
        embedding_depth = self.depth_firstConv(depth.clone())
        corr = self.sigmoid(torch.sum(embedding_rgb * embedding_depth, 1).unsqueeze(1))
        fuse_fea = self.relu(self.fuse_Conv(torch.cat((embedding_rgb, embedding_depth), dim=1)))
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
        self.Conv2 = nn.Conv2d(channel, channel, 3, 1, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // 16, channel, 1, bias=False)

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

class FusionAttention(nn.Module):
    def __init__(self, in_channel = 64):
        super(FusionAttention, self).__init__()
        # temporal attention (before fusion conv)
        self.max_pool = nn.MaxPool2d(3, 2, 1)
        self.avg_pool = nn.AvgPool2d(3, 2, 1)
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
        self.relu = nn.ReLU(inplace=True)
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
        fuse_out = self.relu(self.fuse_outConv(fuse_fea))

        attn_pre_1 = self.lrelu(self.attn_ScaleConv1(fuse_fea))
        attn_max_1 = self.max_pool(attn_pre_1)
        attn_avg_1 = self.avg_pool(attn_pre_1)
        attn_sum_1 = self.lrelu(self.attnConv1(torch.cat((attn_max_1, attn_avg_1), dim=1)))
        attn_sum_1 = self.upsample(attn_sum_1)


        attn_sum_1_out = self.attn_lastConv1(attn_sum_1)
        # attn_sum_2_up = self.attn_lastConv2(self.Up(attn_sum_2))
        attn = torch.sigmoid(attn_sum_1_out)

        out = fuse_out * attn + fuse_out
        return out

class PixUpBlock(nn.Module):
    def __init__(self, in_channel):
        super(PixUpBlock, self).__init__()
        self.up = nn.PixelShuffle(2)
        self.conv = BasicBlock(in_channel//4, in_channel//4)
    def forward(self, x):
        out = self.conv(self.up(x))
        return out

class PixAndUnsample(nn.Module):
    def __init__(self, in_c, out_c):
        super(PixAndUnsample, self).__init__()
        self.up1 = nn.PixelShuffle(2)
        self.conv1 = ConvBlock(in_c // 4, out_c)
        self.up2 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = ConvBlock(in_c , out_c)
        self.sum = ConvBlock(out_c, out_c)
    def forward(self, x):
        x1 = self.conv1(self.up1(x))
        x2 = self.conv2(self.up2(x))
        out = self.sum(x1 + x2)
        return out

class MSJCA(nn.Module):
    def __init__(self):
        super(MSJCA, self).__init__()
        self.rgb_con1 = ResNet34_1()
        self.rgb_con2 = ResNet34_2()
        self.rgb_con3 = ResNet34_3()
        self.rgb_con4 = ResNet34_4()
        self.rgb_con5 = ResNet34_5()

        self.dep_con1 = ResNet34_1()
        self.dep_con2 = ResNet34_2()
        self.dep_con3 = ResNet34_3()
        self.dep_con4 = ResNet34_4()
        self.dep_con5 = ResNet34_5()

        self.JA1 = JointAttention(in_channel=64, ratio=16)
        self.JA2 = JointAttention(in_channel=64, ratio=16)
        self.JA3 = JointAttention(in_channel=128, ratio=16)
        self.JA4 = JointAttention(in_channel=256, ratio=16)
        self.JA5 = JointAttention(in_channel=512, ratio=16)

    def forward(self, rgb_input, depth_input):
        rgb_1 = self.rgb_con1(rgb_input)
        depth_1 = self.dep_con1(depth_input)
        rgb_1, depth_1 = self.JA1(rgb_1, depth_1)

        rgb_2 = self.rgb_con2(rgb_1)
        depth_2 = self.dep_con2(depth_1)
        rgb_2, depth_2 = self.JA2(rgb_2, depth_2)

        rgb_3 = self.rgb_con3(rgb_2)
        depth_3 = self.dep_con3(depth_2)
        rgb_3, depth_3 = self.JA3(rgb_3, depth_3)

        rgb_4 = self.rgb_con4(rgb_3)
        depth_4 = self.dep_con4(depth_3)
        rgb_4, depth_4 = self.JA4(rgb_4, depth_4)

        rgb_5 = self.rgb_con5(rgb_4)
        depth_5 = self.dep_con5(depth_4)
        rgb_5, depth_5 = self.JA5(rgb_5, depth_5)

        ## remember 3,4,5
        return rgb_3, rgb_4, rgb_5, depth_3, depth_4, depth_5

class Decoder(nn.Module):
    def __init__(self, c1=512, c2=256, c3=128, c4=64, c5=32, c6=16, center_c=4):
        super(Decoder, self).__init__()
        self.B1 = BasicBlockC(c1, c1)
        self.Up1 = PixAndUnsample(c1, c2)
        self.B2 = BasicBlockC(c2*2, c2)
        self.Up2 = PixAndUnsample(c2, c3)
        self.B3 = BasicBlockC(c3*2, c3)
        self.Up3 = PixAndUnsample(c3, c4)
        self.B4 = BasicBlockC(c4, c4)
        self.Up4 = PixAndUnsample(c4, c5)
        self.centerConv = BasicBlockC(center_c, c5)
        self.fusionConv = ConvBlock(c5*2, c5)
        self.B5 = BasicBlockC(c5, c5)
        self.Up5 = PixAndUnsample(c5, c6)
        self.B6 = BasicBlockC(c6, c6)
    def forward(self, x1, x2, x3, center):
        #center = 4, 128, 128
        x1_out = self.B1(x1) # 512, 8, 8
        x1_up = self.Up1(x1_out) # 256, 16, 16
        x2_out = self.B2(torch.cat((x2, x1_up), dim=1)) # 256, 16, 16
        x2_up = self.Up2(x2_out) # 128, 32, 32
        x3_out = self.B3(torch.cat((x3, x2_up), dim=1)) # 128, 32, 32
        x3_up = self.Up3(x3_out) # 64, 64, 64
        x4_out = self.B4(x3_up) # 64, 64, 64
        x4_up = self.Up4(x4_out) # 32, 128, 128
        center_out = self.centerConv(center) # 32, 128, 128
        fusion_fea = self.fusionConv(torch.cat((x4_up, center_out), dim=1)) # 32, 128, 128
        x5_out = self.B5(fusion_fea) # 32, 128, 128
        x5_up = self.Up5(x5_out) # 16, 256, 256
        out = self.B6(x5_up) # 16, 256, 256
        return out


# 可能的总体模型
class MSSOD(nn.Module):
    def __init__(self, rfb_out_channel=32):
        super(MSSOD, self).__init__()
        self.MSJCA = MSJCA()

        self.CCA = CenterCombineAttention(in_channel=256)
        # self.CenterUpConvLast = nn.Sequential(
        #     PixUpBlock(8),
        #     nn.Conv2d(2, 2, 3, 1, 1)
        # )

        self.rgbUpblock1 = nn.Sequential(
            PixUpBlock(128),
        )

        self.rgbUpblock2 = nn.Sequential(
            PixUpBlock(256),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            PixUpBlock(128)
        )

        self.rgbUpblock3 = nn.Sequential(
            PixUpBlock(512),
            PixUpBlock(128),
            nn.Conv2d(32, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            PixUpBlock(128),
        )
        self.depthUpblock1 = nn.Sequential(
            PixUpBlock(128),
        )

        self.depthUpblock2 = nn.Sequential(
            PixUpBlock(256),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            PixUpBlock(128)
        )

        self.depthUpblock3 = nn.Sequential(
            PixUpBlock(512),
            PixUpBlock(128),
            nn.Conv2d(32, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            PixUpBlock(128),
        )

        self.fattn1 = FusionAttention(128)
        self.fattn2 = FusionAttention(256)
        self.fattn3 = FusionAttention(512)

        self.Decoder = Decoder()

        self.last_conv = nn.Sequential(
            BasicBlockC(16, 8),
            BasicBlockC(8, 4),
            BasicBlockC(4, 2),
        )
        self.out_conv = nn.Conv2d(2, 1, 3, 1, 1)

        # self.Decoder = nn.Sequential()

    def forward(self, low_input, high_input):
        # VGG
        rgb_1, rgb_2, rgb_3, depth_1, depth_2, depth_3 = self.MSJCA(low_input, high_input)
        # print("pre:")
        # print(rgb_1.size())
        # print(rgb_2.size())
        # print(rgb_3.size())
        # print(depth_1.size())
        # print(depth_2.size())
        # print(depth_3.size())
        # pre:
        # torch.Size([4, 128, 32, 32])
        # torch.Size([4, 256, 16, 16])
        # torch.Size([4, 512, 8, 8])

        # torch.Size([4, 128, 32, 32])
        # torch.Size([4, 256, 16, 16])
        # torch.Size([4, 512, 8, 8])
        center = self.CCA(rgb_2, depth_2)
        rd_1 = self.fattn1(rgb_1, depth_1)
        rd_2 = self.fattn2(rgb_2, depth_2)
        rd_3 = self.fattn3(rgb_3, depth_3)

        de_out = self.Decoder(rd_3, rd_2, rd_1, center)

        out = self.out_conv(self.last_conv(de_out))

        print("out:", out.size())
        return out

