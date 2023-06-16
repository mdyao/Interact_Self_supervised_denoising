import numpy as np
from PIL import Image
import os
import math
import lpips
import torch
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt
import torchvision
import glob
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def read_image(path):
    return np.asfarray(Image.open(path))/255


def cal_psnr(img1, img2):
    diff = img1 - img2
    rmse = math.sqrt(np.mean(diff**2))
    return 20 * math.log10(1.0 / rmse)

def cal_lpips(img1,img2,normalize=True):
    # img1 = torch.from_numpy(np.transpose(img1, (2,0,1))[None,::].astype(np.float32)).to(device)
    # img2 = torch.from_numpy(np.transpose(img2, (2,0,1))[None,::].astype(np.float32)).to(device)
    img1 = torch.from_numpy(img1[None,None,::].astype(np.float32)).to(device)
    img2 = torch.from_numpy(img2[None,None,::].astype(np.float32)).to(device)
    lpips = loss_fn_alex(img1, img2,normalize=normalize)
    return lpips.item()

def cal_ssim(image1, image2, multichannel=False):
    return compare_ssim(image1, image2, multichannel=multichannel)


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize
        self.loss_fn = torch.nn.MSELoss(reduction='elementwise_mean')

    def forward(self, input, target):



        # if input.shape[1] != 3:
        #     input = input.repeat(1, 3, 1, 1)
        #     target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        # if self.resize:
        #     input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        #     target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            # loss += torch.nn.functional.l1_loss(x, y)
            loss += self.loss_fn(x, y)
        return loss

loss_vgg = VGGPerceptualLoss().to(device)
loss_fn_alex = lpips.LPIPS(net='alex').to(device) # best forward scores

def cal_vgg(img1, img2):
    # image1 = torch.from_numpy(np.transpose(image1, (2,0,1))[None,::].astype(np.float32)).to(device)
    # image2 = torch.from_numpy(np.transpose(image2, (2,0,1))[None,::].astype(np.float32)).to(device)
    img1 = torch.from_numpy(img1[None,None,::].astype(np.float32)).to(device)
    img2 = torch.from_numpy(img2[None,None,::].astype(np.float32)).to(device)
    vgg=loss_vgg(img1, img2)
    return vgg.item()

folder = r'./pred0_1'

gt_list = sorted(glob.glob(os.path.join(r'/ssd1/vis/yaomingde/VDN/data/BSD68','*.png')))
# list1 = sorted(glob.glob(os.path.join(folder,'*000.png')))+
createVar = locals()
for i in np.arange(2.1,3.1,0.1):
    createVar['list'+ str(int(i*10))] = sorted(glob.glob(os.path.join(folder,'*%03d.png'%(i*10))))

def cal_list_metric(gt_list, list1):
    psnr_list = []
    ssim_list = []
    vgg_list = []
    lpips_list = []
    for i in tqdm(range(len(gt_list)),ncols=45):
        gt = read_image(gt_list[i])
        gt = gt[:gt.shape[0]//16*16,:gt.shape[1]//16*16]
        img = read_image(list1[i])

        psnr_list += [cal_psnr(img,gt)]
        ssim_list += [cal_ssim(img,gt)]
        vgg_list += [cal_vgg(img,gt)]
        lpips_list += [cal_lpips(img,gt)]
    return np.mean(psnr_list),np.mean(ssim_list), np.mean(vgg_list), np.mean(lpips_list)

for i in np.arange(2.1,3.1,0.1):
    content = 'list' + str(int(i*10))
    print(i, cal_list_metric(gt_list, eval(content)))


