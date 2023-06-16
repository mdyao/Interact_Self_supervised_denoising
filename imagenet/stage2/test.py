# from __future__ import print_function

import os
import torch.nn.parallel
import torch.utils.data
import argparse
import numpy as np
from tqdm import tqdm
from utils.saver import Saver
from utils.metrics import Evaluator
from utils.util import *
from PIL import Image
import math
from math import log10
from torchvision import transforms
from dataset import *
from models.model_lib.InterDn import *
import cv2
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='N2V_oneframe')

# Data specifications

parser.add_argument('--dataset', type=str, default='CBSD68')
parser.add_argument('--train_dir', type=str, default=r'blend_gaussian10-50')
parser.add_argument('--train_dir_pred', type=str, default=r'blend_gaussian10-50')
parser.add_argument('--valid_dir', type=str, default=r'IL_test')
parser.add_argument('--workers', type=int, default=0,
                    metavar='N', help='dataloader threads')
# training hyper params
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: auto)')
parser.add_argument('--start_epoch', type=int, default=0,
                    metavar='N', help='start epochs (default:0)')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--patch_size', type=int, default=128)

# optimizer params
parser.add_argument('--lr_G', type=float, default=4e-4, metavar='LR',
                    help='learning rate (default: auto)')
# checking point
parser.add_argument('--resume', type=str, default='model.pth.tar',
                    help='put the path to resuming file if needed')
parser.add_argument('--checkname', type=str, default='Unet',
                    help='set the checkpoint name')
# evaluation option
parser.add_argument('--eval_interval', type=int, default=1,
                    help='evaluation interval (default: 1)')
parser.add_argument('--output_path', type=str, default='./pred0_1',
                    help='evaluuation interval (default: 1)')
args = parser.parse_args()

class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Define Saver
        self.saver = Saver(args)
        self.writer = self.saver.create_summary()

        # Define Dataloader
        self.transform_train = transforms.Compose(
            [ RandomFlip(), ToTensor()])
        self.transform_val = transforms.Compose(
            [ ToTensor()])

        self.transform_inv = transforms.Compose([ToNumpy()])

        self.dataset_train = Dataset(args.train_dir,args.train_dir_pred, data_type='float32', transform=self.transform_train, sgm=25, ratio=0.9,randomcrop=(self.args.patch_size,self.args.patch_size),
                             size_window=(5, 5))
        self.dataset_val = Dataset_train_val(args.valid_dir, data_type='float32', transform=self.transform_val, sgm=25, ratio=0.9,
                             size_window=(5,5))

        self.loader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=args.batch_size, shuffle=True,drop_last=True, num_workers=self.args.workers)
        self.loader_val = torch.utils.data.DataLoader(self.dataset_val, batch_size=1, shuffle=False, num_workers=self.args.workers)

        self.num_train = len(self.dataset_train)
        self.num_val = len(self.dataset_val)

        self.num_batch_train = int((self.num_train / args.batch_size) + ((self.num_train % args.batch_size) != 0))
        self.num_batch_val = int((self.num_val / args.batch_size) + ((self.num_val % args.batch_size) != 0))

        # Define network
        self.netG = InterDnNet(3,3)

        init_net(self.netG, init_type='normal', init_gain=0.02, gpu_ids=[0])

        self.fn_REG = nn.L1Loss().to(device)  # Regression loss: L1

        enc_dict = ['enc1_1.cbr.0.conv.weight', 'enc1_1.cbr.0.conv.bias', 'enc1_2.cbr.0.conv.weight',
                         'enc1_2.cbr.0.conv.bias', 'enc2_1.cbr.0.conv.weight', 'enc2_1.cbr.0.conv.bias',
                         'enc2_2.cbr.0.conv.weight', 'enc2_2.cbr.0.conv.bias', 'enc3_1.cbr.0.conv.weight',
                         'enc3_1.cbr.0.conv.bias', 'dec3_1.decbr.0.deconv.weight', 'dec3_1.decbr.0.deconv.bias',
                         'dec2_2.decbr.0.deconv.weight', 'dec2_2.decbr.0.deconv.bias', 'dec2_1.decbr.0.deconv.weight',
                         'dec2_1.decbr.0.deconv.bias', 'dec1_2.decbr.0.deconv.weight', 'dec1_2.decbr.0.deconv.bias',
                         'dec1_1.decbr.0.deconv.weight', 'tail.0.cbr.0.conv.weight', 'tail.0.cbr.0.conv.bias',
                         'tail.1.cbr.0.conv.weight', 'tail.1.cbr.0.conv.bias']
        # Define Optimizer
        for name, p in self.netG.named_parameters():
            if name in enc_dict:
                p.requires_grad = False

        self.paramsG = self.netG.parameters()
        self.optimG = torch.optim.Adam(filter(lambda x: x.requires_grad is not False, self.netG.parameters()),
                                       lr=args.lr_G, betas=(0.5, 0.999))

        # Clear start epoch if fine-tuning
        self.saver.print_log(args)
        self.saver.print_log('Starting Epoch: {}'.format(args.start_epoch))
        self.saver.print_log('Total Epoches: {}'.format(args.epochs))

        enc_dict=[]
        model_now_dict = self.netG.state_dict()
        pretrained_dict = torch.load(os.path.join(args.resume))
        pretrained_dict = pretrained_dict['state_dict']
        new_state_dict = {k: v for k, v in pretrained_dict.items() if
                          k not in enc_dict}
        model_now_dict.update(new_state_dict)
        self.netG.load_state_dict(model_now_dict)

        self.best_psnr = 0
    def training(self, epoch):

        ## training phase
        self.netG.train()

        loss_l1_train =[]
        loss_G_train = []

        for iter, data in enumerate(tqdm(self.loader_train, ncols=45)):
            label = data['label'].to(device)
            input = data['input'].to(device)
            mask1 = data['mask'].to(device)
            data_pred = data['clean'].to(device)

            # alpha = torch.from_numpy((np.random.rand(1) * 0.2).astype(np.float32)).to(device)
            # lambd = np.random.choice(a=[1,2], size=1, replace=False, p=None)[0]
            # if lambd ==1:
            #     alpha=0.8
            # else:
            #     alpha = 1
            # # forward netG
            # alpha=0.1
            # input = alpha * data_pred + (1-alpha) * input
            output1 = self.netG(input*mask1, 1)

            # backward netG
            self.optimG.zero_grad()

            loss_G1 = self.fn_REG(output1 * (1 - mask1), label * (1 - mask1))

            loss_G = loss_G1

            loss_G.backward()
            self.optimG.step()

            # get losses
            loss_l1_train += [loss_G1.item()]
            loss_G_train += [loss_G.item()]

            self.saver.print_log('TRAIN: EPOCH {}: ITER {:0>4d}/{:0>4d}: LOSS: {:.4f}'
                  .format(epoch, iter, self.num_batch_train, np.mean(loss_G_train)))

        # if epoch % self.args.eval_interval == 0 or epoch == args.epochs and epoch!=0:
        self.writer.add_scalar('train/l1_loss',  np.mean(loss_l1_train), epoch)
        self.writer.add_scalar('train/total_loss',  np.mean(loss_G_train),  epoch )


    def valid(self, epoch):
        ## validation phase
        np.random.seed(0)

        with torch.no_grad():
            self.netG.eval()
            psnr_list = []

            for iter, data in enumerate(tqdm(self.loader_val, ncols=45)):

                input = data['input'].to(device)
                clean = data['clean'].to(device)
                # data_pred = data['data_pred'].to(device)

                # forward netG
                # alpha = 0.8
                # input = alpha * input + (1-alpha) * data_pred
                # for lambda_c in np.arange(1.1,2.0,0.1):

                # for alpha in np.arange(0,0.11,0.01):
                # alpha=0.1
                #     output1 = alpha * input + (1-alpha) * output
                for lambda0 in [0]:
                    output = self.netG(input, lambda0)

                    output1_grad = output.data.cpu().numpy()[0,:,:,:].transpose(1,2,0)
                    output1_grad = np.clip(output1_grad*255,0,255)
                    Image.fromarray(output1_grad.astype(np.uint8)).save(args.output_path+'/test%03dlambda%03d'%(iter+1, lambda0*100)+'.png')

            #     rmse = np.sqrt(torch.mean(((clean)-(output)).cpu()**2))
            #     psnr = 20*math.log10(1.0/rmse)
            #     psnr_list += [psnr]
            # print(np.mean(psnr_list))
            # if epoch%200==0 and epoch != 0:
            #     self.saver.save_checkpoint(
            #         {'epoch': epoch,
            #     'state_dict': self.netG.state_dict(),
            #     # 'optimizer': self.optimizer.state_dict(),
            #     }, filename = 'checkpoint_%03d.pth.tar'%epoch)
            #     self.saver.print_log('PSNR!!! EPOCH:{:0>3d} PSNR {}'.format(epoch, np.mean(psnr_list)))
            #
            # if epoch>3 and np.mean(psnr_list)> self.best_psnr:
            #     self.best_psnr = np.mean(psnr_list)
            #     self.saver.save_checkpoint(
            #         {'epoch': epoch,
            #     'state_dict': self.netG.state_dict(),
            #     # 'optimizer': self.optimizer.state_dict(),
            #     }, filename = 'checkpoint_best.pth.tar')
            #     self.saver.print_log('BETTER PSNR!!! EPOCH:{:0>3d} PSNR {}'.format(epoch, self.best_psnr))
        np.random.seed(None)

def main():

    trainer=Trainer(args)
    trainer.valid(0)

    # for epoch in tqdm(range(args.start_epoch, args.epochs), ncols=45):
    #     # if epoch % args.eval_interval == 0 or epoch == args.epochs:
    #     trainer.valid(epoch)
    #     trainer.training(epoch)
    trainer.writer.close()

if __name__ == '__main__':
    main()
