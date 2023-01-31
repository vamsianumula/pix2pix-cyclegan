import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class DownSampleBlock(nn.Module):
  def __init__(self, in_ch, out_ch, use_batchnorm=False, stride=2, padding=1):
    super().__init__()
    self.conv1 = nn.Conv2d(in_ch, out_ch, 4,stride=stride, padding=padding, bias=False)
    nn.init.normal_(self.conv1.weight, mean=0.0, std=0.02)
    self.bn = nn.BatchNorm2d(out_ch) if use_batchnorm else None
    self.relu  = nn.LeakyReLU()
  
  def forward(self, x):
    x = self.conv1(x)
    if self.bn:
      x = self.bn(x)
    x = self.relu(x)
    return x
  
class UpSampleBlock(nn.Module):
  def __init__(self, in_ch, out_ch, use_dropout=True):
    super().__init__()
    self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, 4,stride=2, padding=1,bias=False)
    nn.init.normal_(self.conv1.weight, mean=0.0, std=0.02)
    self.bn = nn.BatchNorm2d(out_ch)
    self.relu  = nn.ReLU()
    self.dropout = nn.Dropout(0.5) if use_dropout else None
  
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn(x)
    if self.dropout:
      x = self.dropout(x)
    x = self.relu(x)
    return x

class Encoder(nn.Module):
  def __init__(self, chs, batch_norm):
    super().__init__()
    self.enc_blocks = nn.ModuleList([DownSampleBlock(chs[i], chs[i+1],batch_norm[i]) for i in range(len(chs)-1)])
  
  def forward(self, x):
    ftrs = []
    for block in self.enc_blocks:
        x = block(x)
        ftrs.append(x)
        # print(x.shape)
    return x,ftrs

class Decoder(nn.Module):
  def __init__(self, chs, dropout):
    super().__init__()
    self.dec_blocks = nn.ModuleList([UpSampleBlock(2*chs[i], chs[i+1],dropout[i]) for i in range(len(chs)-1)])
    self.dec_blocks[0]=UpSampleBlock(chs[0],chs[1])
  
  def forward(self, x, encoder_features):
    for block, ftr in zip(self.dec_blocks,encoder_features):
        x = block(x)
        x = torch.cat([x, ftr], dim=1)
    return x

class UnetGenerator(nn.Module):
  def __init__(self, lambdaa=100):
    super().__init__()
    self.enc_chs = [3,64,128,256,512,512,512,512,512]
    self.enc_bn = [True]*len(self.enc_chs)
    self.enc_bn[0]=False
    self.encoder     = Encoder(self.enc_chs,self.enc_bn)
    
    self.dec_chs = self.enc_chs[::-1]
    self.dec_dropout = [False]*len(self.enc_chs)
    self.dec_dropout[0:3]=[True]*3
    self.decoder     = Decoder(self.dec_chs,self.dec_dropout)
    
    self.head        = nn.ConvTranspose2d(128,3, 4,stride=2, padding=1,bias=False)
    nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
    self.head_act   = nn.Tanh()
    self.lambdaa = lambdaa 

  def forward(self, x):
    x, ftrs = self.encoder(x)
    ftrs.reverse()
    out      = self.decoder(x, ftrs[1:])
    out      = self.head_act(self.head(out))
    return out