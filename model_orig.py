import torch
import torch.nn as nn


class pconv(nn.Module):
  
  # Here, Mask is assumed to be 1 where missing!!!

  def __init__(self,in_channels,out_channels,kernel_size,stride):
    super(pconv,self).__init__()
    self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding=1)
    self.mask_conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding=1,bias=False)
    nn.init.constant_(self.mask_conv.weight.data, val=1) # All ones

    # Disable gradient update on the mask kernel
    for param in self.mask_conv.parameters():
      param.requires_grad=False
  
  def forward(self,x,m):
    # Step 1 : Apply input convolution to the masked input
    h = self.conv(x*m)
    # Step 2 : Assign by reference bias separately (just len channel)
    bias = self.conv.bias.view(1,-1,1,1).expand(h.shape[0],-1,h.shape[2],h.shape[3])
    # Step 3 : Apply pooling on the mask and make non zero entries all 1
    with torch.no_grad():
      m_sum = self.mask_conv(m)
    non_zero = m_sum > 0
    zero = m_sum == 0
    m_out = m_sum.masked_fill_(non_zero,1.0)

    # Step 4 : Apply the final operation (W.T(x*m) - b(x*m))*1/(M)+b
    y = (h-bias)/m_sum + bias
    y = y.masked_fill_(zero,0.0)


    return y,m_sum 

class conv_module(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size,stride,bn=True,act='relu'):
    super(conv_module,self).__init__()
    self.conv = pconv(in_channels,out_channels,kernel_size,stride)
    if bn:
      self.batch_norm = nn.BatchNorm2d(out_channels)

    if act == 'relu':
      self.act = nn.ReLU()
    elif act == 'leaky':
      self.act = nn.LeakyReLU()
  
  def forward(self,x,m):
    # Apply convolution
    h,m_out = self.conv(x,m)
    if hasattr(self,'batch_norm'):
      h = self.batch_norm(h)   
    if hasattr(self,'act'):
      h = self.act(h)
    return h,m_out


class pconv_net(nn.Module):
  def __init__(self,in_features,out_features,base_channel):
    super(pconv_net,self).__init__()
    #Encoder blocks
    self.encoder1 = conv_module(in_features,base_channel,3,2)
    self.encoder2 = conv_module(base_channel,base_channel*2,3,2)
    self.encoder3 = conv_module(base_channel*2,base_channel*4,3,2)
    self.encoder4 = conv_module(base_channel*4,base_channel*8,3,2)

    #Decoder blocks (the channel concat & forward conv module done at once)
    self.decoder4 = conv_module(base_channel*8+base_channel*4,base_channel*4,3,1,act='leaky')
    self.decoder3 = conv_module(base_channel*4+base_channel*2,base_channel*2,3,1,act='leaky')
    self.decoder2 = conv_module(base_channel*2+base_channel,base_channel,3,1,act='leaky')
    self.decoder1 = conv_module(base_channel+in_features,out_features,3,1,act=None)

    # Upsample
    self.usample = nn.Upsample(scale_factor=2,mode='nearest')
  def forward(self,x,m):
    # Define a dictionary of outputs (h,m)
    out_dict = {}

    # First, save the inputs
    out_dict['e0'] = (x,m)

    # Encoder pass
    for i in range(1,5):
      out_dict['e{:d}'.format(i)] = getattr(self,'encoder{:d}'.format(i))(*(out_dict['e{:d}'.format(i-1)]))
    
    h,m_out = out_dict['e{:d}'.format(4)]
    # Decoder pass
    for i in range(4,0,-1):
      # Upsample h and m
      h = self.usample(h)
      m_out = self.usample(m_out)

      

      # Concatenate 
      h_enc,m_enc = out_dict['e{:d}'.format(i-1)]
      h = torch.concatenate([h,h_enc],dim=1)
      m_out = torch.concatenate([m_out,m_enc],dim=1)

      # Pass through decoder
      h,m_out = getattr(self,'decoder{:d}'.format(i))(h,m_out)
    
    return h


      




    

