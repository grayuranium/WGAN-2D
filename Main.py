# ！/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/10 上午10:28
# @Author  : Ryu
# @Site    : 
# @File    : Main.py
# @Software: PyCharm

from wgan import Generator
from wgan import Discriminator
import numpy as np
import os
import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

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
ndf = 16

#Use GPU
cuda = True if torch.cuda.is_available() else False

#Create dictionary
train_data_path = "./data/cifar10/train"
test_data_path = "./data/cifar10/test"
img_save_path = "./result/wgan/cifar10/train"
model_save_path = "./log/wgan/cifar10"
os.makedirs(train_data_path,exist_ok=True)
os.makedirs(test_data_path,exist_ok=True)
os.makedirs(img_save_path,exist_ok=True)
os.makedirs(model_save_path,exist_ok=True)

#Load data
my_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
])
train_dataset = datasets.CIFAR10(train_data_path,train=True,download=True,transform=my_trans)
test_dataset = datasets.CIFAR10(test_data_path,train=False,download=True,transform=my_trans)
train_dataloader = DataLoader(train_dataset,batch_size=batchs,shuffle=True)
# test_dataloader = DataLoader(test_dataset,batch_size=batch,shuffle=False)

#Initialize generator and discriminator
generator = Generator(latent,img_shape)
discriminator = Discriminator(img_shape)
if cuda:
    generator.cuda()
    discriminator.cuda()

#Optimizers
optimizer_G = optim.RMSprop(generator.parameters(),lr=learning_rate)
optimizer_D = optim.RMSprop(discriminator.parameters(),lr=learning_rate)

#Training
score_tr = np.zeros((epochs,2))
batch_done = 0
total_errG = 0
total_errD = 0
count = 0
for epoch in range(epochs):
    for i, data in enumerate(train_dataloader, 1):
        #Train discriminator
        img, label = data
        if cuda:
            img.cuda()
        #Sample noise as generator input
        z = torch.randn([img.size(0), latent])
        if cuda:
            z.cuda()
        #Forward and get loss
        real_imgs_d = discriminator(img)
        D_x = real_imgs_d.data.mean()

        fake_imgs = generator(z)
        fake_imgs_d = discriminator(fake_imgs.detach())
        D_G_z1 = fake_imgs_d.data.mean()

        errD = -torch.mean(real_imgs_d)+torch.mean(fake_imgs_d)
        total_errD += errD.data.item()

        #Backward
        optimizer_D.zero_grad()
        errD.backward()
        optimizer_D.step()

        #Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-clip_value,clip_value)

        if i%critic_per==0:
            #Train generator every critic_per iterations
            optimizer_G.zero_grad()

            #Forward and get loss
            gen_imgs = generator(z)
            D_G_z2 = gen_imgs.data.mean()
            errG = -torch.mean(discriminator(gen_imgs))
            total_errG+=errG.data.item()
            count+=1

            #Backward
            errG.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D(x):%.4f] [D(G(x)):%.4f/%.4f]"
                % (epoch, epochs, batch_done % len(train_dataloader), len(train_dataloader), errD.item(), errG.item(),D_x,D_G_z1,D_G_z2)
            )

            if batch_done % sample_show == 0:
                save_image(gen_imgs.data[:25], img_save_path+"/%d.png" % batch_done, nrow=5, normalize=True)

            batch_done+=critic_per

    score_tr[epoch,0] = total_errG/count
    score_tr[epoch,1] = total_errD/len(train_dataloader)
    count = 0

#Save model
torch.save(generator.state_dict(), model_save_path+'/netG_.pth')
torch.save(discriminator.state_dict(), model_save_path+'/netD_.pth')
np.save(model_save_path+'/score_tr.npy',score_tr)