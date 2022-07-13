import torch
import numpy

a = torch.rand(4,4,64)
b = torch.rand(4,4,64)

if __name__ == '__main__':
    #c = torch.matmul(a,b)
    c =a*b
    print(c.size())