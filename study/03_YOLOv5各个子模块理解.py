import torch.nn as nn
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.common import *
import torch

def _Focus():
    act = nn.SiLU()
    X = torch.rand(2, 3, 100, 100)
    focus = Focus(3, 64, 3, 1)
    r1 = focus(X)
    print(r1.shape)

def _BottleneckCSP():
    X = torch.rand(2, 64, 100, 100)
    csp = BottleneckCSP(c1=64, c2=64, n=2)
    r1 = csp(X)
    print(r1.shape)

    c3 = C3(c1=64, c2=64, n=2)
    r2 = c3(X)
    print(r2.shape)

def _SPP():
    X = torch.rand(2, 64, 100, 100)

    spp = SPP(c1=64, c2=64)
    r1 = spp(X)
    print(r1.shape)

    sppf = SPPF(c1=64, c2=64)
    sppf.cv1 = spp.cv1
    sppf.cv2 = spp.cv2

    r2 = sppf(X)
    print(r2.shape)

    print(torch.max(torch.abs(r1 - r2)))



if __name__=='__main__':
    # _Focus()
    # _BottleneckCSP()
    _SPP()