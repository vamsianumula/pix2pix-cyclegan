import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from unet_generator import DownSampleBlock

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.down1 = DownSampleBlock(6,64,False)
    self.down2 = DownSampleBlock(64,128,True)
    self.down3 = DownSampleBlock(128,256,True)
    self.down4 = DownSampleBlock(256,512,True,stride=1, padding=1)
    self.conv = nn.Conv2d(512, 1, 4,stride=1, padding=1, bias=False)
    nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
    # self.out = nn.Sigmoid()

  def forward(self, input, target):
    x = torch.cat([input,target],dim=1)
    x =self.down1(x)
    x= self.down2(x)
    x = self.down3(x)
    x = self.down4(x)
    x = self.conv(x)
    # x = self.out(x)
    return x
  
class DownSampleBlockInstance(nn.Module):
  def __init__(self, in_ch, out_ch, use_batchnorm=False, stride=2, padding=1):
    super().__init__()
    self.conv1 = nn.Conv2d(in_ch, out_ch, 4,stride=stride, padding=padding, bias=False)
    nn.init.normal_(self.conv1.weight, mean=0.0, std=0.02)
    self.bn = nn.InstanceNorm2d(out_ch) if use_batchnorm else None
    self.relu  = nn.LeakyReLU()
  
  def forward(self, x):
    x = self.conv1(x)
    if self.bn:
      x = self.bn(x)
    x = self.relu(x)
    return x

class DiscriminatorInstance(nn.Module):
  def __init__(self):
    super().__init__()
    self.down1 = DownSampleBlockInstance(3,64,False)
    self.down2 = DownSampleBlockInstance(64,128,True)
    self.down3 = DownSampleBlockInstance(128,256,True)
    self.down4 = DownSampleBlockInstance(256,512,True,stride=1, padding=1)
    self.conv = nn.Conv2d(512, 1, 4,stride=1, padding=1, bias=False)
    nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
    # self.out = nn.Sigmoid()

  def forward(self, input):
    x =self.down1(input)
    x= self.down2(x)
    x = self.down3(x)
    x = self.down4(x)
    x = self.conv(x)
    # x = self.out(x)
    return x
