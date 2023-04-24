import torch 
import torch.nn as nn
from torchvision.models import vgg16
"""
USAGE OF PERCEPTUAL LOSS 

We will use the pretrained VGG16 but with frozen layers (only used for forward)
We only will use the convolutional part (vgg16.features())
The following blocks of VGG16 are defined

Block 1 : Layers 0-3
Block 2 : Layers 4-8
Block 3 : Layers 9-15
Block 4 : Layers 16-22

"""

class perceptual_loss(nn.Module):
  
  def __init__(self):
    
    self.vgg_conv = vgg16(pretrained=True).features
    
    p = [0,4,9,16,23]
    self.blocks = [self.vgg_conv[p[i]:p[i+1]] for i in range(len(p)-1)]

    # Define average and std as side variable (buffers are not part of model parameters)
    # These parameters were already set using the 
    self.register_buffer(name='mean', tensor = torch.Tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
    self.register_buffer(name='sigma', tensor = torch.Tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

  def forward(self,Igt,Ipred,blocks = [0],norm=False):
    
    if norm:
      Igt = (Igt - self.mean)/self.sigma
      Ipred = (Ipred - self.mean)/self.sigma
    
    