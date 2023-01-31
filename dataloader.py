from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import torchtext
import torchvision
import torchvision.transforms.functional as tf
import numpy as np

class CustomDataset(Dataset):
  def __init__(self, img_dir, data="train", transform=True):
    self.transform = transform
    self.img_dir= os.path.join(img_dir, data)

  def __len__(self):
    return len(os.listdir(self.img_dir)) 
  
  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, f"{idx+1}.jpg")
    img = read_image(img_path,torchvision.io.image.ImageReadMode.RGB)
    width = img.shape[2]//2
    input = img[:,:,width:]
    real = img[:,:,:width]
    if self.transform:
        input, real = self.transform_imgs(input,real,(286,286))
    input = (input / 127.5) - 1
    real = (real / 127.5) - 1
    return input.type('torch.FloatTensor'), real.type('torch.FloatTensor')
  
  def transform_imgs(self,input, real, resize_dim):
    org_dim = (input.shape[1], input.shape[2]) 
    resize = transforms.Resize(size=resize_dim)
    input,real = resize(input), resize(real)
    i, j, h, w = transforms.RandomCrop.get_params(input, output_size=org_dim)
    input, real = tf.crop(input, i, j, h, w), tf.crop(real, i, j, h, w)
    if np.random.rand() > 0.5:
        input,real = tf.hflip(input),tf.hflip(real)
    return input, real

def get_paired_data(config):
    print("Downloading data..")
    torchtext.utils.download_from_url("http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz")
    torchtext.utils.extract_archive("./.data/facades.tar.gz","data")
    
    train_data = CustomDataset(img_dir="data/facades", data="train", transform=True)
    test_data = CustomDataset(img_dir="data/facades", data="test", transform=False)
    #batch_size=4
    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=True)
    
    return train_dataloader, test_dataloader