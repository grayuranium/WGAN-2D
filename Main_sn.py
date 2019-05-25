# ！/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/12 下午8:15
# @Author  : Ryu
# @Site    : 
# @File    : Main_sn.py
# @Software: PyCharm

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

epochs = 200
batchs = 64
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
ndf = 32
b1 = 0.5
b2 = 0.99

#Use GPU
cuda = True if torch.cuda.is_available() else False

#Create dictionary
train_data_path = "./data/mnist/train"
test_data_path = "./data/mnist/test"
img_save_path = "./result/sngan/mnist/train"
model_save_path = "./log/sngan/mnist"
os.makedirs(train_data_path,exist_ok=True)
os.makedirs(test_data_path,exist_ok=True)
os.makedirs(img_save_path,exist_ok=True)
os.makedirs(model_save_path,exist_ok=True)

#Load data
my_trans = transforms.Compose([
    transforms.Resize(ndf),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
])
train_dataset = datasets.MNIST(train_data_path,train=True,download=True,transform=my_trans)
# test_dataset = datasets.MNIST(test_data_path,train=False,download=True,transform=my_trans)
train_dataloader = DataLoader(train_dataset,batch_size=batchs,shuffle=True,num_workers=int(2))
# test_dataloader = DataLoader(test_dataset,batch_size=batch,shuffle=False)

#Initialize generator and discriminator
generator = Generator(latent,img_shape,ndf)
discriminator = Discriminator(img_shape,ndf)
if cuda:
    generator.cuda()
    discriminator.cuda()

#Optimizers
optimizer_G = optim.Adam(generator.parameters(),lr=learning_rate,betas=(b1,b2))
optimizer_D = optim.Adam(discriminator.parameters(),lr=learning_rate,betas=(b1,b2))

#Loss function
adversarial_loss = nn.BCELoss()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

#Training
score_tr = np.zeros((epochs,2))
batch_done = 0
total_errG = 0
total_errD = 0
for epoch in range(epochs):
    for i, data in enumerate(train_dataloader, 1):
        img, label = data
        #Sample real image
        if cuda:
            img.cuda()
        #Sample noise as generator input
        z = torch.randn([img.size(0), latent ,1 ,1])
        if cuda:
            z.cuda()

        # Train generator
        optimizer_G.zero_grad()

        # Forward and get loss
        gen_imgs = generator(z)
        gen_imgs_d = discriminator(gen_imgs)
        D_G_z2 = gen_imgs.data.mean()
        errG = -torch.mean(gen_imgs_d)
        total_errG += errG.data.item()

        # Backward
        errG.backward()
        optimizer_G.step()

        # Train discriminator
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

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D(x):%.4f] [D(G(x)):%.4f/%.4f]"
            % (epoch, epochs, batch_done % len(train_dataloader), len(train_dataloader), errD.item(), errG.item(),D_x,D_G_z1,D_G_z2)
        )

        batch_done = epoch * len(train_dataloader) + i

        if batch_done % sample_show == 0:
            save_image(gen_imgs.data[:25], img_save_path+"/%d.png" % batch_done, nrow=5, normalize=True)

    score_tr[epoch,0] = total_errG/len(train_dataloader)
    score_tr[epoch,1] = total_errD/len(train_dataloader)

#Save model
torch.save(generator.state_dict(), model_save_path+'/netG_.pth')
torch.save(discriminator.state_dict(), model_save_path+'/netD_.pth')