# Variational information distillation (VID) for accelerating Wasserstein GAN

This repository consists of code for an extra experiment for our [CVPR paper](https://arxiv.org/abs/1904.05835).

## The main idea of VID

The idea is to introduce a data-dependent regularizer for deep neural network training.
The regularizer is defined by the mutual information between the activations of the teacher network and the student network, 
where the teacher network is pretrained (possibly on another dataset). 

## An application to GAN

One difficulty in GAN is that the discriminator is not informative enough 
when the generated samples are clearly different from real samples.
A pretrained discriminator obtained from another image generation task
may not be able to tell whether a sample is real but it should be able 
to detect most of fake samples, since it has seen so many fake samples previously.
By leveraging the teacher's discriminator, the objective for the student's discriminator additionally contains the term \(- \lambda I(D_s; D_t)\)
The mutual information is in general intractable. Please check our [CVPR paper](https://arxiv.org/abs/1904.05835) for the approximate implementation.

## Experiments on CIFAR10

The teacher's discriminator 

>Conv2d(128) -> LeakyReLU -> Conv2d(256) -> LeakyReLU -> Conv2d(512) -> LeakyReLU-Linear.

The student's discriminator

>Conv2d(32) -> LeakyReLU -> Conv2d(64) -> LeakyReLU -> Conv2d(128) -> LeakyReLU-Linear.

The mutual information terms are added before each LeakyReLU.
