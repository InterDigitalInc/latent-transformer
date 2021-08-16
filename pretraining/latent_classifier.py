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
os.makedirs(log_dir, exist_ok=True)
logger = Logger(log_dir)

Latent_Classifier = LCNet(fmaps=[9216, 2048, 512, 40], activ='leakyrelu')
Latent_Classifier.to(device)

dataset_A = LatentDataset(opts.latent_path, opts.label_path, training_set=True)
loader_A = data.DataLoader(dataset_A, batch_size=batch_size, shuffle=True)

params = list(Latent_Classifier.parameters())
optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

BCEloss = nn.BCEWithLogitsLoss(reduction='none')
n_iter = 0
for n_epoch in range(epochs):

    scheduler.step()

    for i, list_A in enumerate(loader_A):

        dlatent_A, lbl_A = list_A
        dlatent_A, lbl_A = dlatent_A.to(device), lbl_A.to(device)

        predict_lbl_A = Latent_Classifier(dlatent_A.view(dlatent_A.size(0), -1))
        predict_lbl = F.sigmoid(predict_lbl_A)

        loss = BCEloss(predict_lbl_A, lbl_A.float())
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (n_iter + 1) % 10 == 0:
            logger.log_value('loss', loss.item(), n_iter + 1)
        n_iter += 1

torch.save(Latent_Classifier.state_dict(),'{:s}/latent_classifier_epoch_{:d}.pth'.format(log_dir, n_epoch + 1))

