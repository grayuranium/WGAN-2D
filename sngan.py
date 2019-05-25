# ！/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/12 下午8:14
# @Author  : Ryu
# @Site    : 
# @File    : sngan.py
# @Software: PyCharm

import torch.nn as nn
from function.SNconv import SNConv2d

#out=(in-1)×s+k-2×p
class Generator(nn.Module):
    def __init__(self, input, output, ngf):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # input is (input) x 1 x 1

            nn.ConvTranspose2d(input, ngf * 8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32

            nn.ConvTranspose2d(ngf, output[0], 3, 1, 1, bias=True),
            nn.Tanh()
            # state size. (output[0]) x 32 x 32
        )

    def forward(self, x):
        output = self.model(x)
        return output

#out=(in-k+2×p)/s+1
class Discriminator(nn.Module):
    def __init__(self, input, ndf):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # input is (input[0]) x 28 x 28

            SNConv2d(input[0], ndf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf) x 28 x 28

            SNConv2d(ndf, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf) x 14 x 14

            SNConv2d(ndf, ndf * 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*2) x 14 x 14

            SNConv2d(ndf*2, ndf * 2, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*2) x 7 x 7

            SNConv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*4) x 7 x 7

            SNConv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*8) x 4 x 4

            SNConv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),

            SNConv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        return output.view(-1, 1).squeeze(1)