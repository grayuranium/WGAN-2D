# ！/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/20 下午3:47
# @Author  : Ryu
# @Site    : 
# @File    : ShowPicture.py
# @Software: PyCharm

import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import torch
from visdom import Visdom

from sngan import Generator
from sngan import Discriminator
import numpy as np
import os
import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

batches = 10
learning_rate = 0.00003
cpu = 8
latent = 100
img_size_x = 28
img_size_y = 28
img_size_z = 1
#This is for image_save
img_shape = (img_size_z,img_size_x,img_size_y)
critic_per = 5
clip_value = 0.01
sample_show = 400
b1 = 0.5
b2 = 0.99

epoches = 200
ndf = 152

#Use GPU
cuda = True if torch.cuda.is_available() else False

model_save_path_1 = "./log/wgan/cifar10/score_tr.npy"
model_save_path_2 = "./log/wgan_2d/cifar10/score_tr.npy"

mat_1 = np.load(model_save_path_1)
mat_2 = np.load(model_save_path_2)

mat_G = np.zeros((epoches,2))
mat_D = np.zeros((epoches,2))
for i in range(epoches):
    mat_G[i][0] = mat_1[i][0]
    mat_D[i][0] = mat_1[i][1]
    mat_G[i][1] = -mat_2[i][0]
    mat_D[i][1] = -mat_2[i][1]

viz = Visdom(env='my_window')

#Create loss iteration img
# viz.line(
#     X=np.array(list(range(epoches))),
#     Y=mat_G,
#     opts=dict(
#         showlegend=True,
#         title='生成器迭代Loss变化',
#         legend=['wgan','wgan-2d']
#     )
# )

# Create generated img
# my_transform_1 = transforms.Compose([
# 	# transforms.CenterCrop((224,224)),
#     transforms.Resize(ndf),
# 	transforms.ToTensor(),
# ])
# my_transform_2 = transforms.Compose([
# 	# transforms.CenterCrop((224,224)),
# 	transforms.ToTensor(),
# ])
# img = Image.open('./result/wgan/cifar10/train/155600.png').convert('RGB')
# img_1_cifar10 = my_transform_2(img).unsqueeze(0)
# img = Image.open('./result/wgan/mnist/train/186800.png').convert('RGB')
# img_1_mnist = my_transform_1(img).unsqueeze(0)
# img = Image.open('./result/wgan_dc/cifar10/train/155600.png').convert('RGB')
# img_dc_cifar10 = my_transform_2(img).unsqueeze(0)
# img = Image.open('./result/sngan/cifar10/train/155600.png').convert('RGB')
# img_2_cifar10 = my_transform_2(img).unsqueeze(0)
# img = Image.open('./result/sngan/mnist/train/186800.png').convert('RGB')
# img_2_mnist = my_transform_1(img).unsqueeze(0)
# img_mnist = torch.cat([img_1_mnist,img_2_mnist],dim=0)
# img_cifar10 = torch.cat([img_1_cifar10,img_dc_cifar10,img_2_cifar10],dim=0)
# viz.images(
#     img_mnist,
#     opts=dict(
#         nrow=2,
#         title='MNIST数据集上生成的图片',
#         padding=10,
#         captions=['wgan','wgan-2dsn'],
#     )
# )
# viz.images(
#     img_cifar10,
#     opts=dict(
#         nrow=3,
#         title='CIFAR10数据集上生成的图片',
#         padding=10,
#         captions=['wgan','dcgan','wgan-2dsn'],
#     )
# )

#Show origin foot img
# foot_data_path = './origin_img'
# img_save_path = "./result/sngan/foot/train"
#
# my_trans = transforms.Compose([
#     transforms.Resize(32),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5],[0.5]),
# ])
#
# train_dataset = datasets.ImageFolder(foot_data_path,transform=my_trans)
# train_dataloader = DataLoader(train_dataset,batch_size=batches,shuffle=True,num_workers=int(2))
#
# for epoch in range(2000):
#     for i, data in enumerate(train_dataloader, 1):
#         img, label = data
#         save_image(img.data[:10], img_save_path + "/%d.png" % origin, nrow=5, normalize=True)
#         break
#     break
my_transform_1 = transforms.Compose([
	# transforms.CenterCrop((224,224)),
	transforms.ToTensor(),
])
img = Image.open('./result/sngan/foot/train/origin.png').convert('RGB')
img_real_foot = my_transform_1(img).unsqueeze(0)
img = Image.open('./result/sngan/foot/train/145600.png').convert('RGB')
img_generate_foot = my_transform_1(img).unsqueeze(0)
img_foot = torch.cat([img_real_foot,img_generate_foot],dim=0)
viz.images(
    img_foot,
    nrow=1,
    opts=dict(
        title='WGAN-2D足底压力图像生成',
        padding=10,
        captions=['wgan','wgan-2dsn'],
    )
)