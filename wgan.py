
# coding: utf-8

# In[1]:


import argparse
import os
import numpy as np
import math
import sys
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import tensorboardX
import utils


# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--gpu_id", type=int, default=2, help="gpu id")
parser.add_argument("--thin_factor", type=float, default=0.5, help="thinning generator by a factor")
parser.add_argument("--dir", type=str, default='./', help="directory of each experiment")

#sys.argv = 'main.py'
#sys.argv = sys.argv.split(' ')
opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)
cuda = True if torch.cuda.is_available() else False

img_shape = (opt.channels, opt.img_size, opt.img_size)


# In[3]:


class Generator(nn.Module):
    def __init__(self, thin_factor=1.0):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, int(128*thin_factor), normalize=False),
            *block(int(128*thin_factor), int(256*thin_factor)),
            *block(int(256*thin_factor), int(512*thin_factor)),
            *block(int(512*thin_factor), int(1024*thin_factor)),
            nn.Linear(int(1024*thin_factor), int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


# In[4]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

        utils.init_weights(self.model)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# In[5]:


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# In[6]:


# Configure data loader
os.makedirs("data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)


# In[7]:


# Loss weight for gradient penalty
lambda_gp = 10

# Models
generator = Generator(opt.thin_factor)
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()


# In[8]:


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# TODO: torch.set_default_tensor_type(t) OR device, randn
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# In[9]:


#utils.mkdir(ckpt_dir)
os.makedirs(opt.dir, exist_ok=True)
ckpt_dir = '%s/checkpoints' % opt.dir
os.makedirs(ckpt_dir, exist_ok=True)
try:
    ckpt = utils.load_checkpoint(ckpt_dir)
    start_epoch = ckpt['epoch']
    discriminator.load_state_dict(ckpt['discriminator'])
    generator.load_state_dict(ckpt['generator'])
    optimizer_G.load_state_dict(ckpt['optimizer_G'])
    optimizer_D.load_state_dict(ckpt['optimizer_D'])
except:
    print(' [*] No checkpoint!')
    start_epoch = 0


# In[ ]:


os.makedirs("images", exist_ok=True)
writer = tensorboardX.SummaryWriter('./summaries')


# In[ ]:


batches_done = 0

z_sample = Variable(Tensor(np.random.normal(0, 1, (25, opt.latent_dim))))

for epoch in range(opt.n_epochs):
    tdl = tqdm(iter(dataloader))
    for i, (imgs, targets) in enumerate(tdl):

        step = epoch * len(dataloader) + i + 1

        generator.train()

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        wass_distance = torch.mean(real_validity) - torch.mean(fake_validity)
        d_loss = -wass_distance + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        writer.add_scalar('D/wd', wass_distance.item(), global_step=step)
        writer.add_scalar('D/gp', gradient_penalty.item(), global_step=step)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            writer.add_scalar('G/g_loss', g_loss.item(), global_step=step)

            if batches_done % opt.sample_interval == 0:
                msg = '[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]' % (
                        epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                tdl.set_description(msg)

                generator.eval()
                f_imgs_sample = generator(z_sample)
                #save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                save_image(f_imgs_sample.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += opt.n_critic

    # checkpoint at epoch
    utils.save_checkpoint({'epoch': epoch + 1,
                           'discriminator': discriminator.state_dict(),
                           'generator': generator.state_dict(),
                           'optimizer_D': optimizer_D.state_dict(),
                           'optimizer_G': optimizer_G.state_dict()},
                          '%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1),
                          max_keep=2)

