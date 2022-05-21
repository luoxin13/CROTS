import torch
import kornia
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def color_jitter(colorJitter, img_mean, data = None, target = None, s=0.25):
    # s is the strength of colorjitter
    # colorJitter
    if not (data is None):
        if data.shape[1]==3:
            if colorJitter > 0.2:
                img_mean, _ = torch.broadcast_tensors(img_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3), data)
                seq = nn.Sequential(kornia.augmentation.ColorJitter(brightness=s,contrast=s,saturation=s,hue=s))
                data = (data+img_mean)/255
                data = seq(data)
                data = (data*255-img_mean).float()
    return data, target


def gaussian_blur(blur, data = None, target = None):
    if not (data is None):
        if data.shape[1]==3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(np.floor(np.ceil(0.1 * data.shape[2]) - 0.5 + np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(np.floor(np.ceil(0.1 * data.shape[3]) - 0.5 + np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(kornia.filters.GaussianBlur2d(kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def flip(flip, data = None, target = None):
    #Flip
    if flip == 1:
        if not (data is None): data = torch.flip(data,(3,))#np.array([np.fliplr(data[i]).copy() for i in range(np.shape(data)[0])])
        if not (target is None):
            target = torch.flip(target,(2,))#np.array([np.fliplr(target[i]).copy() for i in range(np.shape(target)[0])])
    return data, target


def cow_mix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        stackedMask, data = torch.broadcast_tensors(mask, data)
        stackedMask = stackedMask.clone()
        stackedMask[1::2]=1-stackedMask[1::2]
        data = (stackedMask*torch.cat((data[::2],data[::2]))+(1-stackedMask)*torch.cat((data[1::2],data[1::2]))).float()
    if not (target is None):
        stackedMask, target = torch.broadcast_tensors(mask, target)
        stackedMask = stackedMask.clone()
        stackedMask[1::2]=1-stackedMask[1::2]
        target = (stackedMask*torch.cat((target[::2],target[::2]))+(1-stackedMask)*torch.cat((target[1::2],target[1::2]))).float()
    return data, target


def mix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        if mask.shape[0] == data.shape[0]:
            data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])
        elif mask.shape[0] == data.shape[0] / 2:
            data = torch.cat((torch.cat([(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))]),
                              torch.cat([((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))])))
    if not (target is None):
        target = torch.cat([(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])
    return data, target


def ms_mix(image, prediction, num_mix=4):
    b_image = image.clone()
    b_pred = prediction.clone()

    if random.random() < 0.25:
        return b_image, b_pred
    
    if b_image.shape[0] > 1:
        idx = random.randint(1, b_image.shape[0]-1)
        f_image = torch.cat((b_image[idx:, :, :, :].clone() , b_image[:idx, :, :, :].clone()), dim=0)
        f_pred = torch.cat((b_pred[idx:, :, :, :].clone() , b_pred[:idx, :, :, :].clone()), dim=0)
    else:
        f_image = f_image.clone()
        f_pred = f_pred.clone()
    for _ in range(num_mix):
        # scale
        scale_factor = 0.5 + random.random() * 1.5
        oh, ow = image.shape[-2:]
        th, tw = round(scale_factor * oh), round(scale_factor * ow)
        f_image = F.interpolate(f_image, size=(th, tw), mode="bicubic", align_corners=False)
        f_pred = F.interpolate(f_pred, size=(th, tw), mode="nearest")

        # flip
        if random.random() < 0.5:
            f_image = torch.flip(f_image,(-2,))
            f_pred = torch.flip(f_pred,(-2,))

        # crop
        crop_scale = 0.2 + random.random() * 0.8
        ch, cw = round(crop_scale * th), round(crop_scale * tw)
        h0 = random.randint(0, th - ch)
        w0 = random.randint(0, tw - cw)
        f_image = f_image[:, :, h0:h0+ch, w0:w0+cw]
        f_pred = f_pred[:, :, h0:h0+ch, w0:w0+cw]

        patch_scale = 0.1 + random.random() * 0.2
        ph, pw = round(patch_scale * oh), round(patch_scale * ow)
        # match patch_size
        f_image = F.interpolate(f_image, size=(ph, pw), mode="bicubic", align_corners=False)
        f_pred = F.interpolate(f_pred, size=(ph, pw), mode="nearest")
        h0 = random.randint(0, oh-ph)
        w0 = random.randint(0, ow-pw)
        # paste
        b_image[:, :, h0:h0+ph, w0:w0+pw] = f_image[:, :, :, :]
        b_pred[:, :, h0:h0+ph, w0:w0+pw] = f_pred[:, :, :, :]

    return b_image, b_pred
