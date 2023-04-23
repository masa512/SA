import torch
import torch.nn as nn

class pconv(nn.Module):
  
  def __init__(self,in_channels,out_channels,kernel_size,stride):
    
    super(self,pconv).__init__()
    
