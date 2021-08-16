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
from torchvision import transforms, utils

class LatentDataset(data.Dataset):
    def __init__(self, latent_dir, label_dir, training_set=True):
        dlatents = np.load(latent_dir)
        labels = np.load(label_dir)

        train_len = int(0.9*len(labels))
        if training_set:
            self.dlatents = dlatents[:train_len] 
            self.labels = labels[:train_len]
            #self.process_score()
        else:
            self.dlatents = dlatents[train_len:]
            self.labels = labels[train_len:]

        self.length = len(self.labels)
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dlatent = torch.tensor(self.dlatents[idx])
        lbl = torch.tensor(self.labels[idx])

        return dlatent, lbl

