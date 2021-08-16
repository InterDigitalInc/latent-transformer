# Copyright (c) 2021, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import argparse
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import yaml

from PIL import Image
from torchvision import transforms, utils, models
from tensorboard_logger import Logger

from datasets import *
from trainer import *
from utils.functions import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='001', help='Path to the config file.')
parser.add_argument('--attr', type=str, default='Eyeglasses', help='attribute for manipulation.')
parser.add_argument('--latent_path', type=str, default='./data/celebahq_dlatents_psp.npy', help='dataset path')
parser.add_argument('--label_file', type=str, default='./data/celebahq_anno.npy', help='label file path')
parser.add_argument('--stylegan_model_path', type=str, default='./pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt', help='stylegan model path')
parser.add_argument('--classifier_model_path', type=str, default='./models/latent_classifier_epoch_20.pth', help='pretrained attribute classifier')
parser.add_argument('--log_path', type=str, default='./logs/', help='log file path')
parser.add_argument('--out_path', type=str, default='./outputs/', help='output path')
opts = parser.parse_args()

# Celeba attribute list
attr_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, \
            'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, \
            'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, \
            'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, \
            'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, \
            'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, \
            'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, \
            'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}


n_steps = 7
scale = 1.5

with torch.no_grad():

    log_dir = os.path.join(opts.log_path, opts.config) + '/'
    config = yaml.load(open('./configs/' + opts.config + '.yaml', 'r'))

    save_dir = opts.out_path + 'test/'
    os.makedirs(save_dir, exist_ok=True)

    attr = opts.attr
    attr_num = attr_dict[attr]

    # Initialize trainer
    trainer = Trainer(config, attr_num, attr, opts.label_file)
    trainer.initialize(opts.stylegan_model_path, opts.classifier_model_path)   
    trainer.load_model(log_dir)
    trainer.to(device)
    
    testdata_dir = './data/test/'
    img_list = [glob.glob1(testdata_dir, ext) for ext in ['*jpg','*png']]
    img_list = [item for sublist in img_list for item in sublist]
    img_list.sort()

    for idx in range(len(img_list)):

        x_0 = img_to_tensor(Image.open(testdata_dir + img_list[idx]))
        x_0 = x_0.unsqueeze(0).to(device)
        img_l = [x_0] # original image

        w_0 = np.load(testdata_dir + 'latent_code_%05d.npy'%idx)
        w_0 = torch.tensor(w_0).to(device)
        predict_lbl_0 = trainer.Latent_Classifier(w_0.view(w_0.size(0), -1))
        lbl_0 = F.sigmoid(predict_lbl_0)
        attr_pb_0 = lbl_0[:, attr_num]
        coeff = -1 if attr_pb_0 > 0.5 else 1   

        range_alpha = torch.linspace(-scale, scale, n_steps)
        for i, alpha in enumerate(range_alpha):

            w_1 = trainer.T_net(w_0.view(w_0.size(0), -1), alpha.unsqueeze(0).to(device))
            w_1 = w_1.view(w_0.size())
            w_1 = torch.cat((w_1[:,:11,:], w_0[:,11:,:]), 1)
            x_1, _ = trainer.StyleGAN([w_1], input_is_latent=True, randomize_noise=False)
            img_l.append(x_1.data)

        out = torch.cat(img_l, dim=3)
        utils.save_image(clip_img(out), save_dir + attr + '_' + '%05d.jpg'%idx)