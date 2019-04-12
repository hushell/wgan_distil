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
parser.add_argument('--dataroot', default='./data', help='where to store datasets')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar', 'stl10'],
                        help='The name of dataset')
parser.add_argument('--download', type=str, default='True', help='whether download')
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--gpu_id", type=int, default=2, help="gpu id")
parser.add_argument("--dir", type=str, default='./', help="directory of each experiment")

#sys.argv = 'main.py'
#sys.argv = sys.argv.split(' ')
opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)
cuda = True if torch.cuda.is_available() else False

channels = 1 if 'mnist' in opt.dataset else 3
img_shape = (channels, opt.img_size, opt.img_size)

#utils.mkdir(ckpt_dir)
ckpt_dir = '%s/checkpoints' % opt.dir
summ_dir = '%s/summaries' % opt.dir
img_dir = '%s/images' % opt.dir

utils.mkdir([opt.dir, ckpt_dir, img_dir, summ_dir])


##########################################################################

DIM = 128 # This overfits substantially; you're probably better off with 64

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(opt.latent_dim, 4 * 4 * 4 * DIM),
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, img_shape[0],
                img_shape[1], img_shape[2])

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
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    interpolates = Variable(interpolates, requires_grad=True)
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


def generate_interpolation():
    z1 = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
    z2 = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
    images = []
    number_int = 10
    for i in range(number_int):
        alpha = i / float(number_int - 1)
        z_intp = z1*alpha + z2*(1.0 - alpha)
        generator_sample = generator(z_intp)
        images.append(generator_sample)
    for i in range(imgs.shape[0]):
        generator_sample = [images[j].data[i] for j in range(number_int)]
        generator_imgs = make_grid(generator_sample, nrow=number_int, normalize=True, scale_each=True)
        save_image(generator_imgs, '%s/interpolate_%d_%d.png' % (img_dir, i, j))
        writer.add_image('interpolat/%d_%d' % (i, j), generator_imgs, global_step=step+1)


##########################################################################
# Configure data loader
dataloader, test_loader = get_data_loader(opt)

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
    ckpt = utils.load_checkpoint(ckpt_dir)
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
writer = tensorboardX.SummaryWriter(summ_dir)

batches_done = 0

z_sample = Variable(Tensor(np.random.normal(0, 1, (25, opt.latent_dim))))

for epoch in range(opt.n_epochs):
    tdl = dataloader
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

        writer.add_scalar('D/wd', wass_distance.data[0], global_step=step)
        writer.add_scalar('D/gp', gradient_penalty.data[0], global_step=step)

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

            writer.add_scalar('G/g_loss', g_loss.data[0], global_step=step)

            # print and save
            if batches_done % opt.sample_interval == 0:
                msg = '[Epoch %d/%d] [Batch %d/%d] [D: %.4f] [G: %.4f]' % (
                        epoch, opt.n_epochs, i, len(dataloader), d_loss.data[0], g_loss.data[0])
                print(msg)

                generator.eval()
                generator_sample = generator(z_sample)
                generator_imgs = make_grid(generator_sample.data[:25], nrow=5, normalize=True)
                save_image(generator_imgs, "%s/%d.png" % (img_dir, batches_done))
                writer.add_image('I/%d' % batches_done, generator_imgs, global_step=step)

            batches_done += opt.n_critic

    # checkpoint at epoch
    utils.save_checkpoint({'epoch': epoch + 1,
                           'discriminator': discriminator.state_dict(),
                           'generator': generator.state_dict(),
                           'optimizer_D': optimizer_D.state_dict(),
                           'optimizer_G': optimizer_G.state_dict()},
                          '%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1),
                          max_keep=2)

