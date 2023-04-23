import torch
import torch.nn as nn
from torch.fft import ifft2,ifftshift
import numpy as np
# Part 1 : MSE

class recon_loss(nn.Module):

  def __init__(self):
    super(recon_loss,self).__init__()
    self.mse_loss = nn.MSELoss()
  def forward(self,Fin,Fpred,M,I):

    # Step 1 : Replace masked-out region with the prediction and leave the rest
    Mprime = torch.ones_like(M)-M
    F = Fin*M + Fpred*Mprime
    F = F[:,0,:,:] + 1j * F[:,1,:,:]

    # Step 2 : Take ifftn-ifftshift on F 
    Ipred = abs(ifft2(ifftshift(F)))

    # Step 3 : Evaluate reconstruction loss
    L = self.mse_loss(Ipred,I)

    return L


class s_loss(nn.Module):

  def __init__(self,lmb):
    super(s_loss,self).__init__()
    self.lmb = lmb
    self.sob0 = torch.Tensor(np.array([[1,2,1],[0,0,0],[-1,-2,-1]])).float().unsqueeze(dim=0).unsqueeze(dim=0).to('cuda')
    self.sob1 = torch.Tensor(np.array([[-1,0,1],[-2,0,2],[-1,0,1]])).float().unsqueeze(dim=0).unsqueeze(dim=0).to('cuda')
    self.mse_loss = nn.MSELoss()
  def forward(self,Fpred,M):
    # We will add contribution from both real and imag part 
    # Part 1 : Mask out the predicted

    Fmask = Fpred*M

    # Part 2 : Evaluate gradient using two fixed kernels
    # * Also need to rescale based on relevant number of pixels
    Gyr = abs(nn.functional.conv2d(Fmask[:,:1,:,:],self.sob0))
    Gxr = abs(nn.functional.conv2d(Fmask[:,:1,:,:],self.sob1))

    Gyi = abs(nn.functional.conv2d(Fmask[:,1:2,:,:],self.sob0))
    Gxi = abs(nn.functional.conv2d(Fmask[:,1:2,:,:],self.sob1))

    # Part 3 : Evaluate MSE 
    L = self.lmb * self.mse_loss((Gyr+Gxr+Gyi+Gxi),torch.zeros_like(Gyr))

    return L 
