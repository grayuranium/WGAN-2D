# ！/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/10 上午10:21
# @Author  : Ryu
# @Site    : 
# @File    : wgan.py
# @Software: PyCharm

import numpy as np
from torch import nn

class Generator(nn.Module):
    def __init__(self,input,output):
        super(Generator, self).__init__()
        self.input = input
        self.output = output
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(input, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(output))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.output)
        return img

class Discriminator(nn.Module):
    def __init__(self,input):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(input)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity