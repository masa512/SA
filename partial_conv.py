import torch
import torch.nn as nn

class partial_conv(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size):
    super(partial_conv,self).__init__()
    self.kernel_size = kernel_size
    self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=kernel_size//2)
    self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

  def forward(self,x,m,eps=1e-10):
    x_masked = x * m
    y = self.conv(x_masked)

    # Pool the mask
    m_pool = self.pool(m)
    y = y/(m_pool+eps)
    
    # update the mask
    m_pool[m_pool>0] = 1

    return y,m_pool

