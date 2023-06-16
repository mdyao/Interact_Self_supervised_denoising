import os
import sys

print('Current work path: ', os.getcwd())
print('The path to the package is: ')

sys.path.insert(0, os.getcwd())  # 把当前路径添加到 sys.path 中

for p in sys.path:
    print(p)

from stage1.models.model_lib.layer import CNR2d,Pooling2d,DECNR2d,UnPooling2d

import torch
import torch.nn as nn
from torch.nn import init

def default_conv(in_channels, out_channels, kernel_size, padding, bias=False, init_scale=0.1):
    basic_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
    nn.init.kaiming_normal_(basic_conv.weight.data, a=0, mode='fan_in')
    basic_conv.weight.data *= init_scale
    if basic_conv.bias is not None:
        basic_conv.bias.data.zero_()
    return basic_conv

def default_Linear(in_channels, out_channels, bias=False):
    basic_Linear = nn.Linear(in_channels, out_channels, bias=bias)
    # nn.init.xavier_normal_(basic_Linear.weight.data)
    nn.init.kaiming_normal_(basic_Linear.weight.data, a=0, mode='fan_in')
    basic_Linear.weight.data *= 0.1
    if basic_Linear.bias is not None:
        basic_Linear.bias.data.zero_()
    return basic_Linear

class TuningBlock(nn.Module):
    def __init__(self, input_size, nch_out):
        super(TuningBlock, self).__init__()
        self.conv0 = default_conv(in_channels=input_size, out_channels=input_size,
                                  kernel_size=3, padding=1, bias=False, init_scale=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = default_conv(in_channels=input_size, out_channels=nch_out,
                                  kernel_size=3, padding=1, bias=False, init_scale=0.1)

    def forward(self, x):
        out = self.conv0(x)
        out = self.relu0(out)
        out = self.conv1(out)
        return out


class GlobalTuningBlockModule(nn.Module):
    def __init__(self, channels=64):
        super(GlobalTuningBlockModule, self).__init__()
        self.num_channels = channels
        # define control variable
        self.control_alpha = nn.Sequential(
            default_Linear(512, 256, bias=False),
            default_Linear(256, 128, bias=False),
            default_Linear(128, channels, bias=False)
        )

    def forward(self, alpha):
        input_alpha = self.control_alpha(alpha)
        return input_alpha

class TuningBlockModule(nn.Module):
    def __init__(self, channels=64, nch_out = 64*2):
        super(TuningBlockModule, self).__init__()
        self.num_channels = channels
        self.nch_out = nch_out
        self.adaptive_alpha = nn.Sequential(
                default_Linear(64, nch_out, bias=False))
                # default_Linear(channels, nch_out, bias=False))

        self.tuning_blocks = TuningBlock(channels,nch_out=nch_out)

    def forward(self, x, alpha):
        tuning_f = self.tuning_blocks(x)
        ad_alpha = self.adaptive_alpha(alpha)
        ad_alpha = ad_alpha.view(-1, self.nch_out, 1, 1)
        return torch.sigmoid(tuning_f * ad_alpha)

class InterDnNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(InterDnNet, self).__init__()
        self.globaltuning_blocks = GlobalTuningBlockModule(channels=nch_ker)

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        """
        Encoder part
        """
        self.act = nn.ReLU()
        self.enc1_1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=[], drop=[])
        self.enc1_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=[], drop=[])
        self.tuning_block_en1 = TuningBlockModule(channels=1 * self.nch_in, nch_out = 1*self.nch_ker)

        self.pool1 = Pooling2d(pool=2, type='avg')

        self.enc2_1 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=[], drop=[])
        self.enc2_2 = CNR2d(2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=[], drop=[])
        self.tuning_block_en2 = TuningBlockModule(channels=1 * self.nch_ker, nch_out = 2*self.nch_ker)

        self.pool2 = Pooling2d(pool=2, type='avg')

        self.enc3_1 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=[], drop=[])
        self.tuning_block_en3 = TuningBlockModule(channels=2 * self.nch_ker, nch_out =4*self.nch_ker)

        """
        Decoder part
        """
        self.dec3_1 = DECNR2d(4 * self.nch_ker,     2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=[], drop=[])
        self.tuning_block_de3 = TuningBlockModule(channels=4 * self.nch_ker, nch_out =2*self.nch_ker)

        self.unpool2 = UnPooling2d(pool=2, type='nearest')

        self.dec2_2 = DECNR2d(2 * 2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=[], drop=[])
        self.dec2_1 = DECNR2d(2 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=[], drop=[])
        self.tuning_block_de2 = TuningBlockModule(channels=4 * self.nch_ker, nch_out =1*self.nch_ker)

        self.unpool1 = UnPooling2d(pool=2, type='nearest')

        self.dec1_2 = DECNR2d(2 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=[], drop=[])
        self.dec1_1 = DECNR2d(1 * self.nch_ker,     1 * self.nch_out, kernel_size=3, stride=1, norm=[]       , relu=[],  drop=[], bias=False)
        self.tuning_block_de1 = TuningBlockModule(channels=2 * self.nch_ker, nch_out =1*self.nch_out)
        #
        self.tail = nn.Sequential(CNR2d(1 * self.nch_out,  1 * self.nch_ker, kernel_size=3, stride=1, norm=[]   , relu=0.0,  drop=[]),
                                  CNR2d(1 * self.nch_ker,  1 * self.nch_out, kernel_size=3, stride=1, norm=[]   , relu=[],  drop=[]))

    def forward(self, x,  input_alpha):
        # control_vector = torch.ones(x.shape[0], 512).cuda() * input_alpha
        control_vector = torch.ones(x.shape[0], 512) * input_alpha

        gloabl_vevtor = self.globaltuning_blocks(control_vector)
        enc1 = self.enc1_2(self.act(self.enc1_1(x)))
        tun_out = self.tuning_block_en1(x=x, alpha= gloabl_vevtor)
        enc1 = enc1 * tun_out
        enc1 = self.act(enc1)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2_2(self.act(self.enc2_1(pool1)))
        tun_out = self.tuning_block_en2(x=pool1, alpha= gloabl_vevtor)
        enc2 = enc2 * tun_out
        enc2 = self.act(enc2)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3_1(pool2)
        tun_out = self.tuning_block_en3(x=pool2, alpha= gloabl_vevtor)
        enc3 = enc3 * tun_out
        enc3 = self.act(enc3)

        """
        Encoder part
        """
        dec3 = self.dec3_1(enc3)
        tun_out = self.tuning_block_de3(x=enc3, alpha= gloabl_vevtor)
        dec3 = dec3 * tun_out
        dec3 = self.act(dec3)

        unpool2 = self.unpool2(dec3)
        cat2 = torch.cat([enc2, unpool2], dim=1)
        dec2 = self.dec2_1(self.act(self.dec2_2(cat2)))
        tun_out = self.tuning_block_de2(x=cat2, alpha= gloabl_vevtor)
        dec2 = dec2 * tun_out
        dec2 = self.act(dec2)

        unpool1 = self.unpool1(dec2)
        cat1 = torch.cat([enc1, unpool1], dim=1)
        dec1 = self.dec1_1(self.act(self.dec1_2(cat1)))
        tun_out = self.tuning_block_de1(x=cat1, alpha= gloabl_vevtor)
        dec1 = dec1 * tun_out

        out = self.tail(x-dec1)

        return out

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


if __name__ =="__main__":
    input = torch.rand(1,3,256,256)#.cuda()
    model = InterDnNet(3, 3, 64, [])#.cuda()
    t=torch.Tensor(1)
    t = t#.cuda()
    out = model(input,t)
    print(out.shape)

    # # print(model.state_dict().keys())
    # # # print(model.parameters())
    # enc_dict = ['enc1_1.cbr.0.conv.weight', 'enc1_1.cbr.0.conv.bias', 'enc1_2.cbr.0.conv.weight',
    #             'enc1_2.cbr.0.conv.bias', 'enc2_1.cbr.0.conv.weight', 'enc2_1.cbr.0.conv.bias',
    #             'enc2_2.cbr.0.conv.weight', 'enc2_2.cbr.0.conv.bias', 'enc3_1.cbr.0.conv.weight',
    #             'enc3_1.cbr.0.conv.bias', 'dec3_1.decbr.0.deconv.weight', 'dec3_1.decbr.0.deconv.bias',
    #             'dec2_2.decbr.0.deconv.weight', 'dec2_2.decbr.0.deconv.bias', 'dec2_1.decbr.0.deconv.weight',
    #             'dec2_1.decbr.0.deconv.bias', 'dec1_2.decbr.0.deconv.weight', 'dec1_2.decbr.0.deconv.bias',
    #             'dec1_1.decbr.0.deconv.weight', 'tail.0.cbr.0.conv.weight', 'tail.0.cbr.0.conv.bias',
    #             'tail.1.cbr.0.conv.weight', 'tail.1.cbr.0.conv.bias']
    # # Define Optimizer
    # for name, p in model.named_parameters():
    #     if name in enc_dict:
    #         p.requires_grad = False
    # #
    # # import torch.optim as optim
    # #
    # # optimizer = optim.Adam(filter(lambda x: x.requires_grad is not False, model.parameters()), lr=0.2)
    # for name, p in model.named_parameters():
    #     print(name,p.requires_grad)
