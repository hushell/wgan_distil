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
from losses import Fit2DHomoGaussianLoss
#from residual_network import resnet18
from data_loader import get_data_loader
from inception_score import get_inception_score
from itertools import chain


##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='./data', help='where to store datasets')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar', 'stl10'],
                        help='The name of dataset')
parser.add_argument('--download', type=str, default='True', help='whether download')
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--gpu_id", type=int, default=2, help="gpu id")
parser.add_argument("--dir", type=str, default='./', help="directory of each experiment")
parser.add_argument("--thin_factor", type=int, default=4, help="DIM // thin_factor")
parser.add_argument("--lambda_distil", type=float, default=1e-3, help="coeff of distilation loss")

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
    def __init__(self, nh=DIM):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Conv2d(3, nh, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(nh, 2*nh, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2*nh, 4*nh, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*nh, 1)
        self.nh = nh

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.nh)
        output = self.linear(output)
        return output


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
        generator_sample = generator_sample.mul(0.5).add(0.5)
        images.append(generator_sample)
    for i in range(imgs.shape[0]):
        generator_sample = [images[j].data[i] for j in range(number_int)]
        #generator_imgs = make_grid(generator_sample, nrow=number_int, normalize=True, scale_each=True)
        generator_imgs = make_grid(generator_sample, nrow=number_int)
        save_image(generator_imgs, '%s/interpolate_%d_%d.png' % (img_dir, i, j))
        writer.add_image('interpolat/%d_%d' % (i, j), generator_imgs, global_step=step+1)


##########################################################################
# Configure data loader
dataloader, test_loader = get_data_loader(opt)

##########################################################################
# Loss weight for gradient penalty
lambda_gp = 10
lambda_distil = opt.lambda_distil

class VID(nn.Module):
    def __init__(self, criterion, thin_factor=2):
        super(VID, self).__init__()

        self.teacher = Discriminator(nh=DIM)
        self.student = Discriminator(nh=DIM//thin_factor)

        # (t, s, tc, sc)
        links = [
           (0, 0, DIM, DIM//thin_factor),
           (2, 2, 2*DIM, 2*DIM//thin_factor),
           (4, 4, 4*DIM, 4*DIM//thin_factor),
        ]

        self.criterion = criterion
        self.links = links

        self.features = []
        self.register_hook()

        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

    def register_hook(self):
        def _hook(module, input, output):
            self.features.append(output)

        teacher_modules = list(self.teacher.main.children())
        student_modules = list(self.student.main.children())

        for i, (t, s, tc, sc) in enumerate(self.links):
            teacher_modules[t].register_forward_hook(_hook)
            student_modules[s].register_forward_hook(_hook)
            self.add_module("criterion{}".format(i), self.criterion(
                input_plane=sc,
                target_plane=tc,
                normalize=False))

    def forward(self, images):
        self.features = []

        out = self.student(images)
        _ = self.teacher(images)

        reg_loss = []
        for i, (feature1, feature2) in enumerate(zip(
            self.features[:int(len(self.features)/2)],
            self.features[int(len(self.features)/2):])):

            _criterion = self._modules.get("criterion{}".format(i))
            reg_loss.append(lambda_distil * _criterion(feature1, feature2))

        return out, reg_loss


# Models
generator = Generator()
utils.init_weights(generator)
#discriminator = Discriminator()
#discriminator = resnet18(num_classes=1, pretrained=True)
discriminator = VID(Fit2DHomoGaussianLoss, opt.thin_factor)

if cuda:
    generator.cuda()
    discriminator.cuda()

try:
    ckpt = utils.load_checkpoint('./CIFAR10_disc/checkpoints')
    start_epoch = 0
    discriminator.teacher.load_state_dict(ckpt['discriminator'])
    #generator.load_state_dict(ckpt['generator'])
except:
    print(' [*] No checkpoint!')
    start_epoch = 0


##########################################################################
# Optimizers

trainable_parameters = [p for p in discriminator.parameters() if p.requires_grad]
for k,v in discriminator.named_parameters():
    if v.requires_grad:
        print(k, v.shape)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(trainable_parameters, lr=opt.lr, betas=(opt.b1, opt.b2))

# TODO: torch.set_default_tensor_type(t) OR device, randn
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


##########################################################################
writer = tensorboardX.SummaryWriter(summ_dir)

batches_done = 0
stride = 1

try:
    z_sample = np.load('%s/z_sample.npy' % ckpt_dir)
    z_sample = torch.from_numpy(z_sample)
    if cuda:
        z_sample = z_sample.cuda()
    z_sample = Variable(z_sample)
    print('Load %s/z_sample.npy' % ckpt_dir)
except:
    z_sample = Variable(Tensor(np.random.normal(0, 1, (25, opt.latent_dim))))
    np.save('%s/z_sample' % ckpt_dir, z_sample.data.cpu().numpy())
    print('Save %s/z_sample.npy' % ckpt_dir)

for epoch in range(opt.n_epochs):
    tdl = tqdm(dataloader)
    for i, (imgs, targets) in enumerate(tdl):

        step = epoch * len(dataloader) + i + 1

        generator.train()
        discriminator.train()
        for p in trainable_parameters:  # reset requires_grad
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
        #real_validity, real_reg_loss = discriminator(real_imgs)
        real_validity = discriminator.student(real_imgs)
        # Fake images
        fake_validity, fake_reg_loss = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator.student, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        wass_distance = torch.mean(real_validity) - torch.mean(fake_validity)
        #d_loss = -wass_distance + lambda_gp * gradient_penalty
        d_loss = -wass_distance + lambda_gp * gradient_penalty + sum(fake_reg_loss)
        #d_loss = -wass_distance + lambda_gp * gradient_penalty + sum(fake_reg_loss) + sum(real_reg_loss)

        d_loss.backward()
        optimizer_D.step()

        writer.add_scalar('D/wd', wass_distance.data[0], global_step=step)
        writer.add_scalar('D/gp', gradient_penalty.data[0], global_step=step)
        writer.add_scalar('D/fake_reg_0', fake_reg_loss[0].data[0], global_step=step)
        writer.add_scalar('D/fake_reg_1', fake_reg_loss[1].data[0], global_step=step)
        writer.add_scalar('D/fake_reg_2', fake_reg_loss[2].data[0], global_step=step)
        #writer.add_scalar('D/real_reg_0', real_reg_loss[0].data[0], global_step=step)
        #writer.add_scalar('D/real_reg_1', real_reg_loss[1].data[0], global_step=step)
        #writer.add_scalar('D/real_reg_2', real_reg_loss[2].data[0], global_step=step)

        # -----------------
        #  Train Generator
        # -----------------

        for p in trainable_parameters:
            p.requires_grad = False  # to avoid computation
        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            teacher_imgs = generator(z)
            fake_imgs = generator(z)

            fake_validity = discriminator.student(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            writer.add_scalar('G/g_loss', g_loss.data[0], global_step=step)

            # print and save
            if batches_done % opt.sample_interval == 0:
                msg = '[Epoch %d/%d] [Batch %d/%d] [D: %.4f] [G: %.4f]' % (
                        epoch, opt.n_epochs, i, len(dataloader), d_loss.data[0], g_loss.data[0])

                if batches_done % (opt.sample_interval * stride) == 0:
                    # Workaround because graphic card memory can't store more than 830 examples in memory for generating image
                    # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
                    # This way Inception score is more correct since there are different generated examples from every class of Inception model
                    sample_list = []
                    for i in range(10):
                        z = Variable(Tensor(np.random.normal(0, 1, (800, opt.latent_dim))))
                        samples = generator(z)
                        sample_list.append(samples.data.cpu().numpy())

                    # Flattening list of list into one list
                    new_sample_list = list(chain.from_iterable(sample_list))
                    #print("Calculating Inception Score over 8k generated images")
                    # Feeding list of numpy arrays
                    inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
                                                          resize=True, splits=10)
                    msg += ' [IS: %.4f]' % inception_score[0]
                    writer.add_scalar('G/inception_score_mean', inception_score[0], global_step=step)
                    writer.add_scalar('G/inception_score_std', inception_score[1], global_step=step)

                #print(msg)
                tdl.set_description(msg)

                generator.eval()
                generator_sample = generator(z_sample)
                generator_sample = generator_sample.mul(0.5).add(0.5)
                generator_imgs = make_grid(generator_sample.data[:25], nrow=5)
                save_image(generator_imgs, "%s/%d.png" % (img_dir, batches_done))
                writer.add_image('I/%d' % batches_done, generator_imgs, global_step=step)

            if batches_done % 1e4 == 1e4 - opt.n_critic:
                stride = min(stride*2, 16)
            batches_done += opt.n_critic


    # checkpoint at epoch
    utils.save_checkpoint({'epoch': epoch + 1,
                           'discriminator': discriminator.state_dict(),
                           'generator': generator.state_dict(),
                           'optimizer_D': optimizer_D.state_dict(),
                           'optimizer_G': optimizer_G.state_dict()},
                          '%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1),
                          max_keep=2)

