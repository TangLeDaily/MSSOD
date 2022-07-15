import torch
import numpy
from rootmodel.model_utils import *
from rootmodel.MS_base_CenterConbine import *

a = torch.rand(4,3,512,512)
b = torch.rand(4,3,512,512)

if __name__ == '__main__':
    msjca = MSSOD()
    center, out = msjca(a, b)
    print(center.size())
    print(out.size())
