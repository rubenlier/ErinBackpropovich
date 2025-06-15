#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import torch.nn as nn
import torch


class VGG19(nn.Module):
    """
     Simplified version of the VGG19 "feature" block
     This module's only job is to return the "feature loss" for the inputs
    """

    def __init__(self, channel_in=3, width=64):
        super(VGG19, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, width, 3, 1, 1)
        self.conv2 = nn.Conv2d(width, width, 3, 1, 1)

        self.conv3 = nn.Conv2d(width, 2 * width, 3, 1, 1)
        self.conv4 = nn.Conv2d(2 * width, 2 * width, 3, 1, 1)

        self.conv5 = nn.Conv2d(2 * width, 4 * width, 3, 1, 1)
        self.conv6 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv7 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv8 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)

        self.conv9 = nn.Conv2d(4 * width, 8 * width, 3, 1, 1)
        self.conv10 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv11 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv12 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)

        self.conv13 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv14 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv15 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv16 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.load_params_()

#so here it is actually extracted!!!! 
    def load_params_(self):
        # Download and load Pytorch's pre-trained weights
        state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
        for ((name, source_param), target_param) in zip(state_dict.items(), self.parameters()):
            target_param.data = source_param.data
            target_param.requires_grad = False

    def feature_loss(self, x):
        return (x[:x.shape[0] // 2] - x[x.shape[0] // 2:]).pow(2).mean()

    def forward(self, x):
        """
        :param x: Expects x to be the target and source to concatenated on dimension 0
        :return: Feature loss
        """
        x = self.conv1(x)
        loss = self.feature_loss(x)
        x = self.conv2(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 64x64

        x = self.conv3(x)
        loss += self.feature_loss(x)
        x = self.conv4(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 32x32

        x = self.conv5(x)
        loss += self.feature_loss(x)
        x = self.conv6(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv7(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv8(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 16x16

        x = self.conv9(x)
        loss += self.feature_loss(x)
        x = self.conv10(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv11(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv12(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 8x8

        x = self.conv13(x)
        loss += self.feature_loss(x)
        x = self.conv14(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv15(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv16(self.relu(x))
        loss += self.feature_loss(x)

        return loss/16


# In[ ]:





# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as vutils
from torch.hub import load_state_dict_from_url

import os
import random
import numpy as np
import math
from IPython.display import clear_output
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import trange, tqdm



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


import torch
import torch.nn as nn
import torch.utils.data


class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3):
        super(ResDown, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, kernel_size, 2, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_out // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 2, kernel_size // 2)

        self.act_fnc = nn.ELU()

    def forward(self, x):
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return self.act_fnc(self.bn2(x + skip))


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2):
        super(ResUp, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_in // 2, kernel_size, 1, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_in // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")

        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.up_nn(x)
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return self.act_fnc(self.bn2(x + skip))


class Encoder(nn.Module):
    """
    Encoder block
    Built for a 3x64x64 image and will result in a latent vector of size z x 1 x 1
    As the network is fully convolutional it will work for images LARGER than 64
    For images sized 64 * n where n is a power of 2, (1, 2, 4, 8 etc) the latent feature map size will be z x n x n

    When in .eval() the Encoder will not sample from the distribution and will instead output mu as the encoding vector
    and log_var will be None
    """
    def __init__(self, channels, ch=64, latent_channels=512):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(channels, ch, 7, 1, 3)
        self.res_down_block1 = ResDown(ch, 2 * ch)
        self.res_down_block2 = ResDown(2 * ch, 4 * ch)
        self.res_down_block3 = ResDown(4 * ch, 8 * ch)
        self.res_down_block4 = ResDown(8 * ch, 16 * ch)
        self.conv_mu = nn.Conv2d(16 * ch, latent_channels, 4, 1)
        self.conv_log_var = nn.Conv2d(16 * ch, latent_channels, 4, 1)
        self.act_fnc = nn.ELU()

    def sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = self.act_fnc(self.conv_in(x))
        x = self.res_down_block1(x)  # 32
        x = self.res_down_block2(x)  # 16
        x = self.res_down_block3(x)  # 8
        x = self.res_down_block4(x)  # 4
        mu = self.conv_mu(x)  # 1
        log_var = self.conv_log_var(x)  # 1

        if self.training:
            x = self.sample(mu, log_var)
        else:
            x = mu

        return x, mu, log_var


class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, channels, ch=64, latent_channels=512):
        super(Decoder, self).__init__()
        self.conv_t_up = nn.ConvTranspose2d(latent_channels, ch * 16, 4, 1)
        self.res_up_block1 = ResUp(ch * 16, ch * 8)
        self.res_up_block2 = ResUp(ch * 8, ch * 4)
        self.res_up_block3 = ResUp(ch * 4, ch * 2)
        self.res_up_block4 = ResUp(ch * 2, ch)
        self.conv_out = nn.Conv2d(ch, channels, 3, 1, 1)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.act_fnc(self.conv_t_up(x))  # 4
        x = self.res_up_block1(x)  # 8
        x = self.res_up_block2(x)  # 16
        x = self.res_up_block3(x)  # 32
        x = self.res_up_block4(x)  # 64
        x = torch.tanh(self.conv_out(x))

        return x 


class VAE(nn.Module):
    """
    VAE network, uses the above encoder and decoder blocks
    """
    def __init__(self, channel_in=3, ch=64, latent_channels=512):
        super(VAE, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation
        (for a 64x64 image this is the size of the latent vector)
        """

        self.encoder = Encoder(channel_in, ch=ch, latent_channels=latent_channels)
        self.decoder = Decoder(channel_in, ch=ch, latent_channels=latent_channels)

    def forward(self, x):
        encoding, mu, log_var = self.encoder(x)
        recon_img = self.decoder(encoding)
        return recon_img, mu, log_var


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


import os
import torch
from torchvision import datasets as Datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Parameters
batch_size = 64
image_size = 64
lr = 1e-4
nepoch = 100
start_epoch = 0
model_name = "CelebA"
load_checkpoint = False
save_dir = os.getcwd()
use_cuda = torch.cuda.is_available()
gpu_indx = 0
device = torch.device(f"cuda:{gpu_indx}" if use_cuda else "cpu")

# Download location (will automatically create and extract here)
dataset_root = os.path.join(save_dir, "celeba_data")

# Transform for CelebA
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Download & prepare CelebA
def get_data_celeba(transform, batch_size, download=False, root="celeba_data"):
    print("Downloading (if needed) and loading CelebA...")

    train_dataset = Datasets.CelebA(root=root, split="train", transform=transform, download=download)
    test_dataset  = Datasets.CelebA(root=root, split="test",  transform=transform, download=download)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("Done loading CelebA.")
    return trainloader, testloader

# Load CelebA
trainloader, testloader = get_data_celeba(transform, batch_size, download=False, root=dataset_root)

# Preview some test images
dataiter = iter(testloader)
test_images, _ = next(dataiter)

plt.figure(figsize=(20, 10))
out = vutils.make_grid(test_images[:8], normalize=True)
plt.imshow(out.numpy().transpose((1, 2, 0)))
plt.axis("off")
plt.show()



# Create the feature loss module
feature_extractor = VGG19().to(device)


#Create VAE network
vae_net = VAE(channel_in=3, ch=64, latent_channels=512).to(device)
# setup optimizer
optimizer = optim.Adam(vae_net.parameters(), lr=lr, betas=(0.5, 0.999))
#Loss function
loss_log = []


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


#Create the save directory if it does note exist
if not os.path.isdir(save_dir + "/Models"):
    os.makedirs(save_dir + "/Models")
if not os.path.isdir(save_dir + "/Results"):
    os.makedirs(save_dir + "/Results")

if load_checkpoint:
    checkpoint = torch.load(save_dir + "/Models/" + model_name + "_" + str(image_size) + ".pt", map_location = "cpu")
    print("Checkpoint loaded")
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    vae_net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint["epoch"]
    loss_log = checkpoint["loss_log"]
else:
    #If checkpoint does exist raise an error to prevent accidental overwriting
    if os.path.isfile(save_dir + "/Models/" + model_name + "_" + str(image_size) + ".pt"):
        raise ValueError("Warning Checkpoint exists")
    else:
        print("Starting from scratch")


# In[ ]:





# In[ ]:





# In[ ]:


import torch.nn.functional as F


def kl(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()






for epoch in trange(start_epoch, nepoch, leave=False):
    vae_net.train()
    for i, (images, _) in enumerate(tqdm(trainloader, leave=False)):
        images = images.to(device)

        recon_img, mu, logvar = vae_net(images)
        #VAE loss
        kl_loss = kl(mu, logvar)
        mse_loss = F.mse_loss(recon_img, images)

        #Perception loss
        feat_in = torch.cat((recon_img, images), 0)
        feature_loss = feature_extractor(feat_in)

        loss = kl_loss + mse_loss + feature_loss

        loss_log.append(loss.item())
        vae_net.zero_grad()
        loss.backward()
        optimizer.step()

    #In eval mode the model will use mu as the encoding instead of sampling from the distribution
    vae_net.eval()
    with torch.no_grad():
        recon_img, _, _ = vae_net(test_images.to(device))
        img_cat = torch.cat((recon_img.cpu(), test_images), 2)

        vutils.save_image(img_cat,
                          "%s/%s/%s_%d.png" % (save_dir, "Results" , model_name, image_size),
                          normalize=True)

        #Save a checkpoint
        torch.save({
                    'epoch'                         : epoch,
                    'loss_log'                      : loss_log,
                    'model_state_dict'              : vae_net.state_dict(),
                    'optimizer_state_dict'          : optimizer.state_dict()

                     }, save_dir + "/Models/" + model_name + "_" + str(image_size) + ".pt")  


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




