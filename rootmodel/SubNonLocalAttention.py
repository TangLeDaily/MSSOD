import torch
import torch.nn as nn

class SubNonLocalAttention(nn.Module):
    def __init__(self, split_dim1=2, split_dim2=3, split_num1=8, split_num2=8, y_channel=128, sum_num = 5):
        super(SubNonLocalAttention, self).__init__()
        self.split_dim1 = split_dim1
        self.split_dim2 = split_dim2
        self.split_num1 = split_num1
        self.split_num2 = split_num2
        self.sum_num = sum_num
        self.Ln = nn.Linear(y_channel, 1)



    def forward(self, x, y):
        B, C, H, W = x.size()
        # input：x.shape = y.shape = [B,C,H,W]
        sp_H_x = x.split(self.split_num1, dim=self.split_dim1)
        sp_HW_x = []
        sp_H_y = y.split(self.split_num1, dim=self.split_dim1)
        sp_HW_y = []
        for i in range(self.split_num1):
            sp_W_x = sp_H_x[i].split(self.split_num2, dim=self.split_dim2)
            sp_HW_x.append(sp_W_x)
            sp_W_y = sp_H_y[i].split(self.split_num2, dim=self.split_dim2)
            sp_HW_y.append(sp_W_y)
        # sp_HW_x 第一维度：num1个H分割组，第二维度：num2个W分割组
        for i in range(self.split_num1):
            for j in range(self.split_num2):
                nowfeature = sp_HW_x[i][j] # B, C, H/num1, W/num2
                now_rep = nowfeature.repeat(1,1,self.split_num1,self.split_num2) # B, C, H, W
                cat_two = torch.cat([y, now_rep], dim=1) # B, 2C, H, W
                cat_two = cat_two.permute(0,2,3,1) # B, H, W, 2C
                afterLn = self.Ln(cat_two) # B, H, W, 1
                afterLn = afterLn.permute(0,3,1,2) # B, 1, H, W

                avgpool = nn.AvgPool2d(kernel_size=(H/self.split_num1, W/self.split_num2), stride=(H/self.split_num1, W/self.split_num2))
                maxpool = nn.MaxPool2d(kernel_size=(H/self.split_num1, W/self.split_num2), stride=(H/self.split_num1, W/self.split_num2))
                sum = avgpool(afterLn) + maxpool(afterLn) # B, 1, num1, num2
                sum_v = sum.view(B, 1, -1)
                B_x = []
                B_y = []
                for k in range(B):
                    B1_x = nowfeature[k, :, :, :]
                    B1_sum_v = sum_v[k, :, :]
                    _, indices = B1_sum_v.topk(self.sum_num, dim=1, largest=True)
                    for l in indices[0, :]:
                        keyx = (l+1)//self.split_num1
                        keyy = l%self.split_num1
                        # l, 0, keyx, keyy 为选出的值
                        # 则 sp_HW_y[keyx][keyy]所代表的的 B, C, H/num1, W/num2 中的 l, C, H/num1, W/num2
                        # 与 sp_HW_x[i][j]所代表的的B, C, H/num1, W/num2 中的 l, C, H/num1, W/num2 相关


        return x, y

import torch
import torch.nn as nn

if __name__ == "__main__":
    avgpool = nn.AvgPool2d(8, stride=8)
    maxpool = nn.MaxPool2d(8, stride=8)
    fc = nn.Linear(128, 1)
    xxx= nn.MaxPool2d()

    a = torch.rand([4,64,256,256])
    b = torch.rand([4,64,256//8,256//8])
    c = b.repeat(1,1,8,8)
    d = torch.cat([a,c],dim=1)
    d = d.permute(0,2,3,1)
    print(d.size())
    d = fc(d)
    d = d.permute(0,3,1,2)

    sum = avgpool(d)+maxpool(d)
    ggg = torch.rand([1,8,8])
    print(ggg)
    fff = ggg.view(1, 64)
    print(fff.size())
    print(fff)
    values, indices = fff.topk(5, dim=1, largest=True)
    print(indices)
    print(values)

    for i in indices[0,:]:
        keyx = (i+1)//8
        keyy = (i)%8

        print(keyx)
        print(keyy)
        print(ggg[0, keyx, keyy])
    # print(fff[0, indices])
