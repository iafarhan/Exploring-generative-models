from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
NOISE_DIM = 96



def sample_noise(batch_size, noise_dim, dtype=torch.float, device='cpu'):
  """
  Generate a PyTorch Tensor of uniform random noise.
  """
  noise = None

  noise =  (-1.-(1.)) * torch.rand(batch_size,noise_dim,dtype=dtype,device=device) + 1.

  return noise



def discriminator():
  """
  Build and return a PyTorch nn.Sequential model implementing the architecture in the notebook.
  """
  model = None

  model = nn.Sequential(
    nn.Linear(in_features=784,out_features=256,bias=True),
    nn.LeakyReLU(0.01),
    nn.Linear(in_features=256,out_features=256,bias=True),
    nn.LeakyReLU(0.01),
    nn.Linear(in_features=256,out_features=1,bias=True)
    )

  return model


def generator(noise_dim=NOISE_DIM):
  """
  Build and return a PyTorch nn.Sequential model implementing the architecture in the notebook.
  """
  model = None
    
  model = nn.Sequential(
    nn.Linear(in_features=noise_dim,out_features=1024),
    nn.ReLU(),
    nn.Linear(in_features=1024,out_features=1024),
    nn.ReLU(),
    nn.Linear(in_features=1024,out_features=784), #map again to image size
    nn.Tanh() # to clip image in range -1,1
  )


  return model  

def discriminator_loss(logits_real, logits_fake):
  """
  Computes the discriminator loss described above.

  """
  loss = None

  size = logits_real.shape[0]
  true_labels = torch.ones_like(logits_real)
  fake_labels = torch.zeros_like(logits_fake)

  real_loss = nn.functional.binary_cross_entropy_with_logits(logits_real,true_labels)
  fake_loss = nn.functional.binary_cross_entropy_with_logits(logits_fake,fake_labels)
  loss = real_loss + fake_loss

  return loss

def generator_loss(logits_fake):
  """
  Computes the generator loss described above.

  """
  loss = None

  #notice the torch.ones rather torch.zeros, we want to make sure 
  #discriminator thinks that it is true data
  labels = torch.ones_like(logits_fake)
  loss = nn.functional.binary_cross_entropy_with_logits(logits_fake,labels)

  return loss

def get_optimizer(model):

  optimizer = None

  optimizer = optim.Adam(model.parameters(),lr=1e-3,betas=(0.5,0.999))

  return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
  """
  Compute the Least-Squares GAN loss for the discriminator.

  """
  loss = None

  true_labels = torch.ones_like(scores_real)
  fake_labels = torch.zeros_like(scores_fake)

  real_loss = nn.functional.mse_loss(scores_real,true_labels)
  fake_loss = nn.functional.mse_loss(scores_fake,fake_labels)
  loss = 0.5*(real_loss + fake_loss)

  return loss

def ls_generator_loss(scores_fake):
  """
  Computes the Least-Squares GAN loss for the generator.

  """
  loss = None

  fake_labels = torch.ones_like(scores_fake)
  loss = 0.5 * nn.functional.mse_loss(scores_fake,fake_labels)

  return loss


def build_dc_classifier():
  """
  Build and return a PyTorch nn.Sequential model for the DCGAN discriminator implementing
  the architecture in the notebook.
  """
  model = None

  model = nn.Sequential(
    nn.Unflatten(1,(1,28,28)),
    #32*24*24
    nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,stride=1),
    nn.LeakyReLU(0.01),
    #32*12*12
    nn.MaxPool2d(kernel_size=2,stride=2),
    #64*8*8
    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1),
    nn.LeakyReLU(0.01),
    #64*4*4
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Flatten(),
    nn.Linear(in_features=1024,out_features=1024),
    nn.LeakyReLU(0.01),
    nn.Linear(in_features=1024,out_features=1)
  )

  return model

class Reshape(nn.Module):
  def __init__(self,shape):
    super(Reshape,self).__init__()
    self.shape=shape
  def forward(self,x):
    return x.view(x.shape[0],*self.shape)

def build_dc_generator(noise_dim=NOISE_DIM):
  """
  Build and return a PyTorch nn.Sequential model implementing the DCGAN generator using
  the architecture described in the notebook.
  """
  model = None
  model = nn.Sequential(
    nn.Linear(in_features=noise_dim,out_features=1024),
    nn.ReLU(),
    nn.BatchNorm1d(num_features=1024),
    nn.Linear(in_features=1024,out_features=7*7*128),\
    nn.ReLU(),
    nn.BatchNorm1d(num_features=7*7*128),
    Reshape((128,7,7)),
    nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(num_features=64),
    nn.ConvTranspose2d(in_channels=64,out_channels=1,kernel_size=4,stride=2,padding=1),
    nn.Tanh(),
    Reshape((784,))
  )

  return model
