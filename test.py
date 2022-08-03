import torch
import numpy

import rootmodel.Base_pool_Center_GNL
from rootmodel import MS_base_CenterConbine
from rootmodel import MS_base
from rootmodel import Base_pool_Center
from rootmodel import MS_base_woCA
from rootmodel import ResNet34_NewUp
from rootmodel import *
from rootmodel.model_utils import *
import thop

a = torch.rand(4,3,256,256)
b = torch.rand(4,3,256,256)

if __name__ == '__main__':
    with torch.no_grad():
        orin_model = MS_base.MSSOD()
        woca_model = MS_base_woCA.MSSOD()
        center_model = MS_base_CenterConbine.MSSOD()
        BPC = Base_pool_Center.MSSOD()
        GNL = ResNet34_NewUp.MSSOD()
        # center, out = msjca(a, b)
        # print(center.size())
        # print(out.size())
        # JA_a = MS_base_CenterConbine.JointAttention(in_channel=512)
        # JA_b = MS_base_CenterConbine.JointAttention(in_channel=256)
        # attn_a = torch.rand(1, 512, 16, 16)
        # attn_b = torch.rand(1, 256, 32, 32)
        o2=GNL(a,b)

        # flops_a, params_a = thop.profile(BPC, (a, b))
        # flops_b, params_b = thop.profile(GNL, (a, b))
        #
        # flops_c, params_c = thop.profile(orin_model, (a, b))
        # flops_d, params_d = thop.profile(woca_model, (a, b))
        # flops_e, params_e = thop.profile(center_model, (a, b))
        #
        # print("Channel=256, H=W=32, JointAttention Params is {}, FLOPs is {}".format(params_b, flops_b))
        # print("Channel=512, H=W=16, JointAttention Params is {}, FLOPs is {}".format(params_a, flops_a))
        # print("Model orin, Params is {}, FLOPs is {}".format(params_c, flops_c))
        # print("Model woCA, Params is {}, FLOPs is {}".format(params_d, flops_d))
        # print("Model center, Params is {}, FLOPs is {}".format(params_e, flops_e))
        # print("model EDVR-M, Params is 3.5M = 3,500,000 , FLOPs is 480G = 480,000,000,000")

