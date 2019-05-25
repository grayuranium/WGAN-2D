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
batchs = 10
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
b1 = 0.5
b2 = 0.99

#Use GPU
cuda = True if torch.cuda.is_available() else False

#Create dictionary
train_data_path = "./data/cifar10/train"
test_data_path = "./data/cifar10/test"
img_save_path = "./result/wgan_2d/cifar10/train"
model_save_path = "./log/wgan_2d/cifar10"
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
train_dataset = datasets.CIFAR10(train_data_path,train=True,download=True,transform=my_trans)
# test_dataset = datasets.MNIST(test_data_path,train=False,download=True,transform=my_trans)
train_dataloader = DataLoader(train_dataset,batch_size=batchs,shuffle=True,num_workers=int(2))
# test_dataloader = DataLoader(test_dataset,batch_size=batch,shuffle=False)

#Initialize generator and discriminator
generator = Generator(latent,img_shape,ndf)
discriminator1 = Discriminator(img_shape,ndf)
discriminator2 = Discriminator(img_shape,ndf)
if cuda:
    generator.cuda()
    discriminator1.cuda()
    discriminator2.cuda()

#Optimizers
optimizer_G = optim.Adam(generator.parameters(),lr=learning_rate,betas=(b1,b2))
optimizer_D1 = optim.Adam(discriminator1.parameters(),lr=learning_rate,betas=(b1,b2))
optimizer_D2 = optim.Adam(discriminator2.parameters(),lr=learning_rate,betas=(b1,b2))

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
discriminator1.apply(weights_init_normal)
discriminator2.apply(weights_init_normal)

#Training
score_tr = np.zeros((epochs,2))
batch_done = 0
total_errG = 0
total_errD = 0
for epoch in range(epochs):
    for i, data in enumerate(train_dataloader, 1):
        img, label = data
        #Sample real image
        img_gauss_mean = torch.mean(img,[0])
        img_gauss_std = torch.Tensor(img_gauss_mean.shape).fill_(1.0)
        img_gauss = torch.normal(mean=img_gauss_mean,std=img_gauss_std)
        img_gauss = torch.unsqueeze(img_gauss,0)
        img_gauss = torch.cat((img_gauss,img_gauss,img_gauss,img_gauss,img_gauss,img_gauss,img_gauss,img_gauss,img_gauss,img_gauss),0)
        real_label = torch.Tensor(img.shape[0],1).fill_(1.0).detach_()
        fake_label = torch.Tensor(img.shape[0],1).fill_(0.0).detach_()
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
        gen_imgs_d = discriminator1(gen_imgs)
        D_G_z2 = gen_imgs.data.mean()
        errG = adversarial_loss(gen_imgs_d,real_label)
        total_errG += errG.data.item()

        # Backward
        errG.backward()
        optimizer_G.step()

        # Train discriminator1
        #Forward and get loss
        real_imgs_d = discriminator1(img)
        D_x = real_imgs_d.data.mean()
        errD_real = adversarial_loss(real_imgs_d,real_label)

        fake_imgs = generator(z)
        fake_gauss_mean = torch.mean(fake_imgs,[0])
        fake_gauss_std = torch.Tensor(fake_gauss_mean.shape).fill_(1.0)
        fake_gauss = torch.normal(mean=fake_gauss_mean,std=fake_gauss_std)
        fake_gauss = torch.unsqueeze(fake_gauss,0)
        fake_gauss = torch.cat((fake_gauss,fake_gauss,fake_gauss,fake_gauss,fake_gauss,fake_gauss,fake_gauss,fake_gauss,fake_gauss,fake_gauss),0)
        fake_imgs_d = discriminator1(fake_imgs.detach())
        D_G_z1 = fake_imgs_d.data.mean()
        errD_fake = adversarial_loss(fake_imgs_d,fake_label)

        errD = (errD_real+errD_fake)/2

        #Train discriminator2
        real_imgs_gauss_d = discriminator2(img_gauss)
        errD_real_gauss = adversarial_loss(real_imgs_gauss_d, real_label)

        fake_imgs_gauss_d = discriminator2(fake_gauss)
        errD_fake_gauss = adversarial_loss(fake_imgs_gauss_d,fake_label)

        errD_gauss = (errD_real_gauss+errD_fake_gauss)/2
        errD_full = errD+errD_gauss
        total_errD += errD_full.data.item()

        #Backward
        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        errD_full.backward()
        optimizer_D1.step()
        optimizer_D2.step()


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
torch.save(discriminator1.state_dict(), model_save_path+'/netD1_.pth')
torch.save(discriminator2.state_dict(), model_save_path+'/netD2_.pth')
np.save(model_save_path+'/score_tr.npy',score_tr)