import torch
import torch.nn as nn

class VGG_conv1(nn.Module):
    def __init__(self):
        super(VGG_conv1, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2    2 layers
    def forward(self,x):
        h = x
        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        h_nopool1 = h
        h = self.pool1(h)
        return h

class VGG_conv2(nn.Module):
    def __init__(self):
        super(VGG_conv2, self).__init__()
        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4   2 layers
    def forward(self,x):
        h = x
        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))
        # h_nopool2 = h
        h = self.pool2(h)
        return h

class VGG_conv3(nn.Module):
    def __init__(self):
        super(VGG_conv3, self).__init__()
        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8   4 layers
    def forward(self,x):
        h = x
        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))
        # h_nopool3 = h
        h = self.pool3(h)
        return h

class VGG_conv4(nn.Module):
    def __init__(self):
        super(VGG_conv4, self).__init__()
        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16      4 layers
    def forward(self,x):
        h = x
        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))
        # h_nopool4 = h
        h = self.pool4(h)
        return h

class VGG_conv5(nn.Module):
    def __init__(self):
        super(VGG_conv5, self).__init__()
        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_3 = nn.ReLU(inplace=True)  # 1/32    4 layers
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16      4 layers
    def forward(self,x):
        h = x
        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        h = self.pool5(h)
        return h

class VGG_conv5_Nopool(nn.Module):
    def __init__(self):
        super(VGG_conv5_Nopool, self).__init__()
        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_3 = nn.ReLU(inplace=True)  # 1/32    4 layers
        # self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16      4 layers
    def forward(self,x):
        # orin: B, C=512, H=16, W=16
        # now: B, C=512, H=32, W=32
        h = x
        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        # h = self.pool5(h)
        return h

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlockC(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None):
        super(BasicBlockC, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        if inplanes != planes:
            self.resconv = nn.Conv2d(inplanes, planes, 1, 1, 0)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.inplanes != self.planes:
            residual = self.resconv(residual)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ConvBlock(nn.Module):
    def __init__(self, inc, outc):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(inc, outc, 3, 1, 1)
        self.bn = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)

        x = self.relu(x_cat + self.conv_res(x))
        return x



class VGG19_conv3(nn.Module):
    def __init__(self):
        super(VGG19_conv3, self).__init__()
        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8   4 layers
    def forward(self,x):
        h = x
        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))
        h = self.relu3_4(self.bn3_4(self.conv3_4(h)))
        h = self.pool3(h)
        return h

class VGG19_conv4(nn.Module):
    def __init__(self):
        super(VGG19_conv4, self).__init__()
        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16      4 layers
    def forward(self,x):
        h = x
        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))
        h = self.relu4_4(self.bn4_4(self.conv4_4(h)))
        h = self.pool4(h)
        return h

class VGG19_conv5(nn.Module):
    def __init__(self):
        super(VGG19_conv5, self).__init__()
        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_3 = nn.ReLU(inplace=True)  # 1/32    4 layers
        self.conv5_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_4 = nn.ReLU(inplace=True)  # 1/32    4 layers
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16      4 layers

    def forward(self, x):
        # orin: B, C=512, H=16, W=16
        # now: B, C=512, H=32, W=32
        h = x
        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        h = self.relu5_4(self.bn5_4(self.conv5_4(h)))
        h = self.pool5(h)
        return h

class VGG19_conv5_Nopool(nn.Module):
    def __init__(self):
        super(VGG19_conv5_Nopool, self).__init__()
        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_3 = nn.ReLU(inplace=True)  # 1/32    4 layers
        self.conv5_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_4 = nn.ReLU(inplace=True)  # 1/32    4 layers
        # self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16      4 layers
    def forward(self,x):
        # orin: B, C=512, H=16, W=16
        # now: B, C=512, H=32, W=32
        h = x
        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        h = self.relu5_4(self.bn5_4(self.conv5_4(h)))
        # h = self.pool5(h)
        return h

class ResNet34_Block(nn.Module):
    def __init__(self, in_c, out_c, s=1):
        super(ResNet34_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, s, 1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        out_1 = self.relu(self.bn1(self.conv1(x)))
        out_2 = self.bn2(self.conv2(out_1))
        return out_2


class ResNet34_1(nn.Module):
    def __init__(self):
        super(ResNet34_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # [B,3,256,256] -> [B,64,64,64]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class ResNet34_2(nn.Module):
    def __init__(self):
        super(ResNet34_2, self).__init__()
        self.layer1 = ResNet34_Block(64, 64)
        self.layer2 = ResNet34_Block(64, 64)
        self.layer3 = ResNet34_Block(64, 64)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        # [B,64,64,64] -> [B,64,64,64]
        x_1 = self.relu(self.layer1(x) + x)
        x_2 = self.relu(self.layer2(x_1) + x_1)
        x_3 = self.relu(self.layer3(x_2) + x_2)
        return x_3

class ResNet34_3(nn.Module):
    def __init__(self):
        super(ResNet34_3, self).__init__()
        self.layer1 = ResNet34_Block(64, 128, s=2)
        self.downconv = nn.Sequential(
            nn.Conv2d(64, 128, 1, 2, 0),
            nn.BatchNorm2d(128)
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = ResNet34_Block(128, 128)
        self.layer3 = ResNet34_Block(128, 128)
        self.layer4 = ResNet34_Block(128, 128)

    def forward(self, x):
        # [B,64,64,64] -> [B,128,32,32]
        x_1 = self.relu(self.layer1(x) + self.downconv(x))
        x_2 = self.relu(self.layer2(x_1) + x_1)
        x_3 = self.relu(self.layer3(x_2) + x_2)
        x_4 = self.relu(self.layer4(x_3) + x_3)
        return x_4

class ResNet34_4(nn.Module):
    def __init__(self):
        super(ResNet34_4, self).__init__()
        self.layer1 = ResNet34_Block(128, 256, s=2)
        self.downconv = nn.Sequential(
            nn.Conv2d(128, 256, 1, 2, 0),
            nn.BatchNorm2d(256)
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = ResNet34_Block(256, 256)
        self.layer3 = ResNet34_Block(256, 256)
        self.layer4 = ResNet34_Block(256, 256)
        self.layer5 = ResNet34_Block(256, 256)
        self.layer6 = ResNet34_Block(256, 256)

    def forward(self, x):
        # [B,128,32,32] -> [B,256,16,16]
        x_1 = self.relu(self.layer1(x) + self.downconv(x))
        x_2 = self.relu(self.layer2(x_1) + x_1)
        x_3 = self.relu(self.layer3(x_2) + x_2)
        x_4 = self.relu(self.layer4(x_3) + x_3)
        x_5 = self.relu(self.layer5(x_4) + x_4)
        x_6 = self.relu(self.layer6(x_5) + x_5)
        return x_6

class ResNet34_5(nn.Module):
    def __init__(self):
        super(ResNet34_5, self).__init__()
        self.layer1 = ResNet34_Block(256, 512, s=2)
        self.downconv = nn.Sequential(
            nn.Conv2d(256, 512, 1, 2, 0),
            nn.BatchNorm2d(512)
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = ResNet34_Block(512, 512)
        self.layer3 = ResNet34_Block(512, 512)

    def forward(self, x):
        # [B,256,16,16] -> [B,512,8,8]
        x_1 = self.relu(self.layer1(x) + self.downconv(x))
        x_2 = self.relu(self.layer2(x_1) + x_1)
        x_3 = self.relu(self.layer3(x_2) + x_2)
        return x_3




class ResNet50_ConvBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c, mid_s=1):
        super(ResNet50_ConvBlock, self).__init__()
        self.con1 = nn.Conv2d(in_c, mid_c, 1, 1, 0)
        self.con2 = nn.Conv2d(mid_c, mid_c, 3, mid_s, 1)
        self.con3 = nn.Conv2d(mid_c, out_c, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_1 = self.relu(self.bn1(self.con1(x)))
        out_2 = self.relu(self.bn2(self.con2(out_1)))
        out_3 = self.bn3(self.con3(out_2))
        return out_3

class ResNet50_1(nn.Module):
    def __init__(self):
        super(ResNet50_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # [B,3,256,256] -> [B,64,64,64]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class ResNet50_2(nn.Module):
    def __init__(self):
        super(ResNet50_2, self).__init__()
        # def __init__(self, in_c, mid_c, out_c, stride=1, down=False, down_inc=1, down_outc=1, down_stride=1):
        self.layer1 = ResNet50_ConvBlock(64, 64, 256)
        self.skip = nn.Conv2d(64, 256, 1, 1, 0)
        self.layer2 = ResNet50_ConvBlock(256, 64, 256)
        self.layer3 = ResNet50_ConvBlock(256, 64, 256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # [B,64,64,64] -> [B,256,64,64]
        out_1 = self.relu(self.layer1(x) + self.skip(x))
        out_2 = self.relu(self.layer2(out_1) + out_1)
        out_3 = self.relu(self.layer3(out_2) + out_2)
        return out_3

class ResNet50_3(nn.Module):
    def __init__(self):
        super(ResNet50_3, self).__init__()
        # def __init__(self, in_c, mid_c, out_c, stride=1, down=False, down_inc=1, down_outc=1, down_stride=1):
        self.layer1 = ResNet50_ConvBlock(256, 128, 512, 2)
        self.skip = nn.Conv2d(256, 512, 1, 2, 0)
        self.layer2 = ResNet50_ConvBlock(512, 128, 512)
        self.layer3 = ResNet50_ConvBlock(512, 128, 512)
        self.layer4 = ResNet50_ConvBlock(512, 128, 512)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # [B,256,64,64] -> [B,512,32,32]
        out_1 = self.relu(self.layer1(x) + self.skip(x))
        out_2 = self.relu(self.layer2(out_1) + out_1)
        out_3 = self.relu(self.layer3(out_2) + out_2)
        out_4 = self.relu(self.layer4(out_3) + out_3)
        return out_4

class ResNet50_4(nn.Module):
    def __init__(self):
        super(ResNet50_4, self).__init__()
        # def __init__(self, in_c, mid_c, out_c, stride=1, down=False, down_inc=1, down_outc=1, down_stride=1):
        self.layer1 = ResNet50_ConvBlock(512, 256, 1024, 2)
        self.skip = nn.Conv2d(512, 1024, 1, 2, 0)
        self.layer2 = ResNet50_ConvBlock(1024, 256, 1024)
        self.layer3 = ResNet50_ConvBlock(1024, 256, 1024)
        self.layer4 = ResNet50_ConvBlock(1024, 256, 1024)
        self.layer5 = ResNet50_ConvBlock(1024, 256, 1024)
        self.layer6 = ResNet50_ConvBlock(1024, 256, 1024)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # [B,512,32,32] -> [B,1024,16,16]
        out_1 = self.relu(self.layer1(x) + self.skip(x))
        out_2 = self.relu(self.layer2(out_1) + out_1)
        out_3 = self.relu(self.layer3(out_2) + out_2)
        out_4 = self.relu(self.layer4(out_3) + out_3)
        out_5 = self.relu(self.layer5(out_4) + out_4)
        out_6 = self.relu(self.layer6(out_5) + out_5)
        return out_6

class ResNet50_5(nn.Module):
    def __init__(self):
        super(ResNet50_5, self).__init__()
        # def __init__(self, in_c, mid_c, out_c, stride=1, down=False, down_inc=1, down_outc=1, down_stride=1):
        self.layer1 = ResNet50_ConvBlock(1024, 512, 2048, 2)
        self.skip = nn.Conv2d(1024, 2048, 1, 2, 0)
        self.layer2 = ResNet50_ConvBlock(2048, 512, 2048)
        self.layer3 = ResNet50_ConvBlock(2048, 512, 2048)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # [B,512,32,32] -> [B,1024,16,16]
        out_1 = self.relu(self.layer1(x) + self.skip(x))
        out_2 = self.relu(self.layer2(out_1) + out_1)
        out_3 = self.relu(self.layer3(out_2) + out_2)
        return out_3