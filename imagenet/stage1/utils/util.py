import numpy as np
import torch
import os
from PIL import Image
import torch.nn.functional as F
from scipy.ndimage import measurements, interpolation
import scipy.io as sio
from scipy.ndimage import filters

def clamp_and_2numpy(tensor):
    tensor =  torch.clamp(tensor, 0, 1)
    return tensor.detach().cpu().numpy()

def clamp_value(tensor):
    return torch.clamp(tensor, 0, 1)

def tensor2numpy(tensor):
    return tensor.detach().cpu().numpy()

def save_image(args, image, iteration,suffix='pred'):
    im_save = np.squeeze(image)
    im_save = np.array(im_save)
    # sio.savemat(os.path.join(args.output_path, args.dataset, args.checkname, 'out_%d%s.mat'%(iteration,suffix)), {'image': im_save})

    im_save = Image.fromarray(im_save.astype(np.uint8))
    im_save.save(os.path.join(args.output_path, args.dataset, args.checkname,'out_%d%s.png'%(iteration,suffix)))

def save_init_images(args,lr,hr,lr_gt):
    save_image(args,lr,0,'lr')
    save_image(args,hr,0,'hr')
    save_image(args,lr_gt,0,'lr_gt')

def ycbcr2rgb(ycbcr1):
    # input image range should be [0,255]
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    ycbcr = ycbcr1.copy()
    shape = ycbcr.shape

    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = ycbcr
    rgb[:,0] -= 16.
    rgb[:,1:] -= 128.
#     np.linalg.inv(m.transpose()
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    rgb=rgb.clip(0, 255).reshape(shape)
    return rgb


def rgb2ycbcr(image1):
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])

    image1 = np.array(image1)
    image = image1.copy()
    shape = image.shape
    if len(shape) == 3:
        image = image.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(image, m.transpose() / 255.)
    ycbcr[:, 0] += 16.
    ycbcr[:, 1:] += 128.
    ycbcr = ycbcr.reshape(shape)
    return ycbcr
#
#
# def ToTensor(image):
#     image = np.expand_dims(image,0)
#     # image = np.expand_dims(image,0)
#
#     image = np.array(image).astype(np.float32).transpose((0,3,1,2))  # whc-->chw
#     image /= 255.0
#     image = torch.from_numpy(image).float()
#     return image