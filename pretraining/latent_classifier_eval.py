# Copyright (c) 2021, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import yaml

from PIL import Image
from torchvision import transforms, utils
from tensorboard_logger import Logger

import sys
sys.path.append(".")
sys.path.append("..")

from datasets import *
from nets import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='1001', help='Path to the config file.')
parser.add_argument('--latent_path', type=str, default='./data/celebahq_dlatents_psp.npy', help='dataset path')
parser.add_argument('--label_path', type=str, default='./data/celebahq_anno.npy', help='label file path')
parser.add_argument('--mapping_layers', type=int, default=3, help='mapping layers num')
parser.add_argument('--fmaps', type=int, default=512, help='fmaps num')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file path')
parser.add_argument('--multigpu', type=bool, default=False, help='use multiple gpus')
opts = parser.parse_args()

attr_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, \
            'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, \
            'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, \
            'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, \
            'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, \
            'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, \
            'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, \
            'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

batch_size = 256
epochs = 20

log_dir = os.path.join(opts.log_path, opts.config) + '/'

def last_model(log_dir, model_name=None):
    if model_name == None:
        files_pth = [i for i in os.listdir(log_dir) if i.endswith('.pth')]
        files_pth.sort()
        return torch.load(log_dir + files_pth[-1])
    else:
        return torch.load(log_dir + model_name)

dataset_A = LatentDataset(opts.latent_path, opts.label_path, training_set=False)
loader_A = data.DataLoader(dataset_A, batch_size=batch_size, shuffle=True)

Latent_Classifier = LCNet(fmaps=[9216, 2048, 512, 40], activ='leakyrelu')
Latent_Classifier.to(device)
Latent_Classifier.load_state_dict(last_model(log_dir, None))
Latent_Classifier.eval()

total_num = dataset_A.__len__()
valid_num = []
real_lbl = []
pred_lbl = []

with torch.no_grad():
    for i, list_A in enumerate(loader_A):

        dlatent_A, lbl_A = list_A
        dlatent_A, lbl_A = dlatent_A.to(device), lbl_A.to(device)

        predict_lbl_A = Latent_Classifier(dlatent_A.view(dlatent_A.size(0), -1))
        predict_lbl = F.sigmoid(predict_lbl_A).round().long()

        real_lbl.append(lbl_A)
        pred_lbl.append(predict_lbl.data)
        valid_num.append((lbl_A == predict_lbl).long())

real_lbl = torch.cat(real_lbl, dim=0)
pred_lbl = torch.cat(pred_lbl, dim=0)
valid_num = torch.cat(valid_num, dim=0)

T_num = torch.sum(real_lbl, dim=0)
F_num = total_num - T_num

pred_T_num = torch.sum(pred_lbl, dim=0)
pred_F_num = total_num - pred_T_num

True_Positive = torch.sum(valid_num * real_lbl, dim=0)
True_Negative = torch.sum(valid_num * (1 - real_lbl), dim=0)

# Recall
recall_T = True_Positive.float()/(T_num.float() + 1e-8)
recall_F = True_Negative.float()/(F_num.float() + 1e-8)

# Precision
precision_T = True_Positive.float()/(pred_T_num.float() + 1e-8)
precesion_F = True_Negative.float()/(pred_F_num.float() + 1e-8)

# Accuracy
valid_num = torch.sum(valid_num, dim=0)
accuracy = valid_num.float()/total_num

for i in range(40):
    print('%0.3f'%recall_T[i].item(), '%0.3f'%recall_F[i].item(), '%0.3f'%precision_T[i].item(), '%0.3f'%precesion_F[i].item(), '%0.3f'%accuracy[i].item())