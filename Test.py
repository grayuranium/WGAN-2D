# ！/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/13 上午6:59
# @Author  : Ryu
# @Site    : 
# @File    : Test.py
# @Software: PyCharm

import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from sngan import Generator
from sngan import Discriminator

from function.metric import compute_score_raw

epochs = 200
batchs = 64
learning_rate = 0.00003
cpu = 8
latent = 100
img_size_x = 32
img_size_y = 32
img_size_z = 3
#This is for image_save
img_shape = (img_size_z,img_size_x,img_size_y)
critic_per = 5
clip_value = 0.01
sample_show = 400
ndf = 32
#help='cifar10 | lsun | imagenet | folder | lfw | fake'
dataset = 'cifar10'
#number of samples for evaluation
sample_size = 2000

#Use GPU
cuda = True if torch.cuda.is_available() else False

#Create dictionary
score_save_path = "./result/wgan_2d/cifar10/test"
test_data_path = "./data/cifar10/test"
model_save_path = "./log/wgan_2d/cifar10"
os.makedirs(score_save_path,exist_ok=True)
os.makedirs(test_data_path,exist_ok=True)
os.makedirs(model_save_path,exist_ok=True)

#Load data
my_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
])
test_dataset = datasets.CIFAR10(test_data_path,train=False,download=True,transform=my_trans)
test_dataloader = DataLoader(test_dataset,batch_size=batchs,shuffle=False,num_workers=int(2))

#Initialize generator and discriminator
generator = Generator(latent,img_shape,ndf)
discriminator = Discriminator(img_shape,ndf)
if cuda:
    generator.cuda()
    discriminator.cuda()

generator.load_state_dict(torch.load(model_save_path+'/netG_.pth',map_location='cpu'))
discriminator.load_state_dict(torch.load(model_save_path+'/netD1_.pth',map_location='cpu'))
generator.eval()

# compute initial score
s = compute_score_raw(dataset, img_size_x, test_data_path, sample_size, batchs, score_save_path+'/real/', score_save_path+'/fake/',
                             generator, latent, conv_model='inception_v3', workers=int(2))

np.save('%s/score.npy' % (score_save_path), s)