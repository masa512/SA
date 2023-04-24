import torch
import torch.nn as nn


class pconv(nn.Module):
  
  # Here, Mask is assumed to be 1 where missing!!!

  def __init__(self,in_channels,out_channels,kernel_size,stride):
    super(self,pconv).__init__()
    self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding=1)
    self.mask_conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding=1,bias=False)
    nn.init.constant_(self.mask_conv.weight.data, val=1) # All ones

    # Disable gradient update on the mask kernel
    for param in self.mask_conv.parameters():
      param.requires_grad=False
  
  def forward(self,x,m):
    # Step 1 : Apply input convolution to the masked input
    h = self.conv(x,m)

    # Step 2 : Assign by reference bias separately (just len channel)
    bias = self.conv.bias.view(1,-1,1,1).expand((h.shape[0],1,h.shape[2],h.shape[3]))

    # Step 3 : Apply pooling on the mask and make non zero entries all 1
    with torch.no_grad():
      m_out = self.mask_conv(m)
    zero = m_out == 0
    m_out.masked_fill_(zero,1.0)

    # Step 4 : Apply the final operation (W.T(x*m) - b(x*m))*1/(M)+b
    y = (h-bias)/m_out + bias
    m = m_out

    return y,m