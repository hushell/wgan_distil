# coding: utf-8

import argparse
import os
import numpy as np
import math
import sys
#from tqdm import tqdm, tqdm_notebook
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

import tensorboardX
import utils
from losses import FitHomoGaussianLoss,Fit2DHomoGaussianLoss,MSELoss
from residual_network import resnet18
from data_loader import get_data_loader


##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='./data')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar', 'stl10'],
                        help='The name of dataset')
parser.add_argument('--download', type=str, default='True')
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
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

#utils.mkdir(ckpt_dir)
ckpt_dir = '%s/checkpoints' % opt.dir
summ_dir = '%s/summaries' % opt.dir
img_dir = '%s/images' % opt.dir

os.makedirs(opt.dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)
os.makedirs(summ_dir, exist_ok=True)


##########################################################################

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

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


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


##########################################################################
# Configure data loader
dataloader, test_loader = get_data_loader(args)

##########################################################################
# Loss weight for gradient penalty
lambda_gp = 10
lambda_distil = 0.01
links = [
   (4, 4, 64, 64),
   (5, 5, 128, 128),
   (6, 6, 256, 256),
   (7, 7, 512, 512)
]

# Models
generator = Generator()
#discriminator = Discriminator()
discriminator = resnet18(num_classes=1, pretrained=True)
utils.init_weights(generator)

if cuda:
    generator.cuda()
    discriminator.cuda()

try:
    ckpt = utils.load_checkpoint('./checkpoints')
    start_epoch = ckpt['epoch']
    discriminator.load_state_dict(ckpt['discriminator'])
    generator.load_state_dict(ckpt['generator'])
    optimizer_D.load_state_dict(ckpt['optimizer_D'])
    optimizer_G.load_state_dict(ckpt['optimizer_G'])
except:
    print(' [*] No checkpoint!')
    start_epoch = 0

# Freeze teacher parameters
#for param in generator.parameters():
#    param.requires_grad = False


##########################################################################
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# TODO: torch.set_default_tensor_type(t) OR device, randn
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


##########################################################################
os.makedirs("images", exist_ok=True)
writer = tensorboardX.SummaryWriter(summ_dir)

batches_done = 0

z_sample = Variable(Tensor(np.random.normal(0, 1, (25, opt.latent_dim))))

for epoch in range(opt.n_epochs):
    tdl = tqdm(dataloader)
    for i, (imgs, targets) in enumerate(tdl):

        step = epoch * len(dataloader) + i + 1

        generator.train()
        discriminator.train()
        for p in discriminator.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

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

        for p in discriminator.parameters():
            p.requires_grad = False  # to avoid computation
        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            teacher_imgs = generator(z)
            fake_imgs = generator(z)

            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            writer.add_scalar('G/g_loss', g_loss.item(), global_step=step)

            # print and save
            if batches_done % opt.sample_interval == 0:
                msg = '[Epoch %d/%d] [Batch %d/%d] [D: %.4f] [G: %.4f]' % (
                        epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                tdl.set_description(msg)

                generator.eval()
                generator_sample = generator(z_sample)
                generator_imgs = make_grid(generator_sample.data[:25], nrow=5)
                save_image(generator_imgs, "%s/%d.png" % (img_dir, batches_done))
                writer.add_image('I/%d' % batches_done, generator_imgs, global_step=step)

            batches_done += opt.n_critic

    # checkpoint at epoch
    utils.save_checkpoint({'epoch': epoch + 1,
                           'discriminator': discriminator.state_dict(),
                           'generator': generator.state_dict(),
                           'optimizer_D': optimizer_D.state_dict(),
                           'optimizer_G': optimizer_G.state_dict(),
                          '%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1),
                          max_keep=2)

