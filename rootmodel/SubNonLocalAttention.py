import torch
import torch.nn as nn
from rootmodel.model_utils import *


# x_conv 和 y_conv 可自定义，预处理卷积层

class SubNonLocalAttention(nn.Module):
    def __init__(self, split_dim1=2, split_dim2=3, split_num1=8, split_num2=8, y_channel=64, sum_num=5):
        super(SubNonLocalAttention, self).__init__()
        self.x_conv = BasicConv2d(y_channel, y_channel, 3)
        self.y_conv = BasicConv2d(y_channel, y_channel, 3)
        self.split_dim1 = split_dim1
        self.split_dim2 = split_dim2
        self.split_num1 = split_num1
        self.split_num2 = split_num2
        self.sum_num = sum_num
        self.Ln = nn.Linear(y_channel * 2, 1)
        self.out_conv = BasicConv2d(y_channel, y_channel, 3)

    def mu(self, x, y):
        C, H, W = x.size()
        a = x.contiguous().view(C, H * W)
        a = a.permute(1, 0)
        b = y.contiguous().view(C, H * W)
        c = a.clone()
        d = torch.matmul(a, b)
        d = torch.matmul(d, c)
        d = d.permute(1, 0).view(C, H, W)
        return d

    def forward(self, x, y):
        B, C, H, W = x.size()
        x = self.x_conv(x)
        y = self.y_conv(y)
        # input：x.shape = y.shape = [B,C,H,W]
        sp_H_x = x.split(H // self.split_num1, dim=self.split_dim1)
        sp_HW_x = []
        sp_H_y = y.split(H // self.split_num1, dim=self.split_dim1)
        sp_HW_y = []
        for i in range(self.split_num1):
            sp_W_x = sp_H_x[i].split(W // self.split_num2, dim=self.split_dim2)
            sp_HW_x.append(sp_W_x)
            sp_W_y = sp_H_y[i].split(W // self.split_num2, dim=self.split_dim2)
            sp_HW_y.append(sp_W_y)
        sp_OutHW_x = []

        for i in range(self.split_num1):
            sp_temp = []
            for j in range(self.split_num2):

                nowfeature = sp_HW_x[i][j]  # B, C, H/num1, W/num2
                now_rep = nowfeature.repeat(1, 1, self.split_num1, self.split_num2)  # B, C, H, W
                cat_two = torch.cat([y, now_rep], dim=1)  # B, 2C, H, W
                cat_two = cat_two.permute(0, 2, 3, 1)  # B, H, W, 2C
                afterLn = self.Ln(cat_two)  # B, H, W, 1
                afterLn = afterLn.permute(0, 3, 1, 2)  # B, 1, H, W

                avgpool = nn.AvgPool2d(kernel_size=(H // self.split_num1, W // self.split_num2),
                                       stride=(H // self.split_num1, W // self.split_num2))
                maxpool = nn.MaxPool2d(kernel_size=(H // self.split_num1, W // self.split_num2),
                                       stride=(H // self.split_num1, W // self.split_num2))
                sum = avgpool(afterLn) + maxpool(afterLn)  # B, 1, num1, num2
                sum_v = sum.view(B, 1, -1)
                Bx_temp = []
                for k in range(B):
                    B1_sum_v = sum_v[k, :, :]
                    _, indices = B1_sum_v.topk(self.sum_num, dim=1, largest=True)
                    x_temp = torch.zeros(1, C, H // self.split_num1, W // self.split_num2)
                    m_x = sp_HW_x[i][j]
                    m_x = m_x[k, :, :, :]
                    for l in indices[0, :]:
                        keyx = l // (self.split_num2)
                        keyy = l % (self.split_num2)
                        m_y = sp_HW_y[keyx][keyy]
                        m_y = m_y[k, :, :, :]
                        x_x = self.mu(m_x, m_y)
                        x_temp = x_temp + x_x
                    Bx_temp.append(x_temp)
                Bx_temp = torch.cat(Bx_temp, dim=0)  # B, C, H/num1, W/num2
                sp_temp.append(Bx_temp)
            sp_OutW_x = torch.cat(sp_temp, dim=self.split_dim2)
            sp_OutHW_x.append(sp_OutW_x)
        out = torch.cat(sp_OutHW_x, dim=self.split_dim1)
        print(out.size())
        out = self.out_conv(out)

        return out


import torch
import torch.nn as nn

if __name__ == "__main__":
    SNLA = SubNonLocalAttention(split_dim1=2, split_dim2=3, split_num1=8, split_num2=2, y_channel=64)
    a = torch.rand([4, 64, 32, 32])
    b = torch.rand([4, 64, 32, 32])
    c = SNLA(a, b)
    # print(fff[0, indices])
