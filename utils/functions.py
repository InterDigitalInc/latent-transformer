# Copyright (c) 2021, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from PIL import Image
from torch.autograd import grad
from torchvision import transforms, utils

        
def clip_img(x):
    """Clip image to range(0,1)"""
    img_tmp = x.clone()[0]
    img_tmp = (img_tmp + 1) / 2
    img_tmp = torch.clamp(img_tmp, 0, 1)
    return [img_tmp.detach().cpu()]

def stylegan_to_classifier(x):
    """Clip image to range(0,1)"""
    img_tmp = x.clone()
    img_tmp = torch.clamp((0.5*img_tmp + 0.5), 0, 1)
    img_tmp = F.interpolate(img_tmp, size=(224, 224), mode='bilinear')
    img_tmp[:,0] = (img_tmp[:,0] - 0.485)/0.229
    img_tmp[:,1] = (img_tmp[:,1] - 0.456)/0.224
    img_tmp[:,2] = (img_tmp[:,2] - 0.406)/0.225
    return img_tmp

def img_to_tensor(x):
    out = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])(x)
    return out


img_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
    
def downscale(x, scale_times=1):
    for i in range(scale_times):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
    return x
    
def upscale(x, scale_times=1):
    for i in range(scale_times):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
    return x
    
def hist_transform(source_tensor, target_tensor):
    """Histogram transformation"""
    c, h, w = source_tensor.size()
    s_t = source_tensor.view(c, -1)
    t_t = target_tensor.view(c, -1)
    s_t_sorted, s_t_indices = torch.sort(s_t)
    t_t_sorted, t_t_indices = torch.sort(t_t)
    for i in range(c):
        s_t[i, s_t_indices[i]] = t_t_sorted[i]
    return s_t.view(c, h, w)

def init_weights(m):
    """Initialize layers with Xavier uniform distribution"""
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.Linear:
        nn.init.uniform_(m.weight, 0.0, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)

def reg_loss(img):
    """Total variation"""
    reg_loss = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))\
             + torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    return reg_loss

def vgg_transform(x):
    """Adapt image for vgg network, x: image of range(0,1) subtracting ImageNet mean"""
    r, g, b = torch.split(x, 1, 1)
    out = torch.cat((b, g, r), dim = 1)
    out = F.interpolate(out, size=(224, 224), mode='bilinear')
    out = out*255.
    return out

def stylegan_to_vgg(x):
    """Clip image to range(0,1)"""
    img_tmp = x.clone()
    img_tmp = torch.clamp((0.5*img_tmp + 0.5), 0, 1)
    img_tmp = F.interpolate(x, size=(224, 224), mode='bilinear')
    img_tmp[:,0] = (img_tmp[:,0] - 0.485)
    img_tmp[:,1] = (img_tmp[:,1] - 0.456)
    img_tmp[:,2] = (img_tmp[:,2] - 0.406)
    r, g, b = torch.split(img_tmp, 1, 1)
    img_tmp = torch.cat((b, g, r), dim = 1)
    img_tmp = img_tmp*255.
    return img_tmp

