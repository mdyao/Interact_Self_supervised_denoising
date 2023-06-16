from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.vgg import vgg16

try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse
import getpass
# user_name = getpass.getuser() # 获取当前用户名

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReconstructionLoss(nn.Module):
    def __init__(self, size_average=True, batch_average=True):
        super(ReconstructionLoss, self,).__init__()
        if torch.cuda.is_available() :#and user_name!='mingd':
            vgg = vgg16(pretrained=False)
            vgg.load_state_dict(torch.load('/gdata/yaomd/pretrained/vgg/vgg16-397923af.pth'))
        else:
            vgg = vgg16(pretrained=True)
        self.loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False
            
        self.size_average = size_average
        self.batch_average = batch_average
        self.L2_loss = nn.MSELoss()

    def L1_Charbonnier_loss(self, X, Y):
        self.eps = 1e-3
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

    def forward(self,X, Y):
        # xx=X.repeat(1,3,1,1)
        # yy=Y.repeat(1,3,1,1)

        #perception_loss = self.L2_loss(self.loss_network(X), self.loss_network(Y))
        mseloss = self.L2_loss(X,Y)
        loss = mseloss#+0.01*perception_loss
        return loss



if __name__ == "__main__":
    loss = ReconstructionLoss()
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 3,7, 7).cuda()
    print(a)

    print(b)
    print(loss(a, b).item())




