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


class VAE(nn.Module):
    def __init__(self, input_size, latent_size=15):
        super(VAE, self).__init__()
        self.input_size = input_size # H*W
        self.latent_size = latent_size # Z
        self.hidden_dim = 256 # H_d
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        self.encoder = nn.Sequential(
          nn.Flatten(start_dim=1,end_dim=-1),
          nn.Linear(in_features=input_size, out_features=self.hidden_dim),
          nn.ReLU(),
          nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
          nn.ReLU(),
          nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
          nn.ReLU()
        )
        self.mu_layer = nn.Linear(in_features = self.hidden_dim, out_features = self.latent_size)
        self.logvar_layer = nn.Linear(in_features = self.hidden_dim, out_features = self.latent_size)

        self.decoder = nn.Sequential(
          nn.Linear(in_features = self.latent_size, out_features = self.hidden_dim),
          nn.ReLU(),
          nn.Linear(in_features=self.hidden_dim,out_features=self.hidden_dim),
          nn.ReLU(),
          nn.Linear(in_features=self.hidden_dim,out_features=self.hidden_dim),
          nn.ReLU(),
          nn.Linear(in_features=self.hidden_dim,out_features=self.input_size),
          nn.Sigmoid(),
          nn.Unflatten(-1,(1,28,28))#1,(C,H,W))
        
        )

  

    def forward(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through 
        encoder, reparametrize trick, and decoder models
        """
        x_hat = None
        mu = None
        logvar = None

        encoder_out = self.encoder(x)
        mu = self.mu_layer(encoder_out)
        logvar = self.logvar_layer(encoder_out)
        z = reparametrize(mu,logvar)
        x_hat = self.decoder(z)

        return x_hat, mu, logvar


class CVAE(nn.Module):
    def __init__(self, input_size, num_classes=10, latent_size=15,hidden_dim=256):
        super(CVAE, self).__init__()
        self.input_size = input_size # H*W
        self.latent_size = latent_size # Z
        self.num_classes = num_classes # C
        self.hidden_dim = hidden_dim # H_d
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None


        enc_in=self.input_size+self.num_classes
        self.encoder = nn.Sequential(
          nn.Linear(in_features=enc_in, out_features=self.hidden_dim),
          nn.ReLU(),
          nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
          nn.ReLU(),
          nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
          nn.ReLU()
        )
        self.mu_layer = nn.Linear(in_features = self.hidden_dim, out_features = self.latent_size)
        self.logvar_layer = nn.Linear(in_features = self.hidden_dim, out_features = self.latent_size)

        dec_in = self.latent_size + self.num_classes
        self.decoder = nn.Sequential(
          nn.Linear(in_features = dec_in, out_features = self.hidden_dim),
          nn.ReLU(),
          nn.Linear(in_features=self.hidden_dim,out_features=self.hidden_dim),
          nn.ReLU(),
          nn.Linear(in_features=self.hidden_dim,out_features=self.hidden_dim),
          nn.ReLU(),
          nn.Linear(in_features=self.hidden_dim,out_features=self.input_size),
          nn.Sigmoid(),
          nn.Unflatten(-1,(1,28,28)))#1,(C,H,W))


    def forward(self, x, c):
        """
        Performs forward pass through FC-CVAE model by passing image through 
        encoder, reparametrize trick, and decoder models
        """
        x_hat = None
        mu = None
        logvar = None

        #N,C = c.shape
        #c = c.view(N,1,1,C)
        X = torch.cat((x.flatten(1,-1),c),1)
        encoder_out = self.encoder(X)
        mu = self.mu_layer(encoder_out)
        logvar= self.logvar_layer(encoder_out)
      
        z = reparametrize(mu,logvar)
        Z = torch.cat((z,c),1)
        x_hat = self.decoder(Z)

        return x_hat, mu, logvar



def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance using the
    reparameterization trick.
    """
    z = None

    N,Z = mu.shape
    z = mu + (torch.exp(logvar)**0.5) * torch.randn(N,Z,device=mu.device,dtype=mu.dtype)

    return z


def loss_function(x_hat, x, mu, logvar):
    """
    Computes the negative variational lower bound loss term of the VAE (refer to formulation in notebook).
    """
    loss = None

    recon_loss = F.binary_cross_entropy(x_hat,x,reduction='sum')
    kl_div = - 0.5 * torch.sum(1. + logvar-mu**2 -logvar.exp())
    loss =  kl_div + recon_loss 
    loss = loss/x.shape[0]

    return loss

