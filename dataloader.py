from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import torchtext
import torchvision
import torchvision.transforms.functional as tf
import numpy as np

class CustomPairedDataset(Dataset):
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

class CustomUnpairedDataset(Dataset):
  def __init__(self, img_dir, data="trainA", transform=True):
    self.transform = transform
    self.img_dir= os.path.join(img_dir, data)
    self.img_list = os.listdir(self.img_dir)

  def __len__(self):
    return len(self.img_list)
  
  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_list[idx])
    input = read_image(img_path,torchvision.io.image.ImageReadMode.RGB)
    if self.transform:
        input = self.transform_img(input,(286,286))
    input = (input / 127.5) - 1
    return input.type('torch.FloatTensor')
  
  def transform_img(self,input, resize_dim):
    org_dim = (input.shape[1], input.shape[2]) 
    resize = transforms.Resize(size=resize_dim)
    input = resize(input)
    i, j, h, w = transforms.RandomCrop.get_params(input, output_size=org_dim)
    input = tf.crop(input, i, j, h, w)
    if np.random.rand() > 0.5:
        input = tf.hflip(input)
    return input

def get_paired_data(config):
    print("Downloading data..")
    torchtext.utils.download_from_url("http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz")
    torchtext.utils.extract_archive("./.data/facades.tar.gz","data")
    
    train_data = CustomPairedDataset(img_dir="data/facades", data="train", transform=True)
    test_data = CustomPairedDataset(img_dir="data/facades", data="test", transform=False)
    #batch_size=4
    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=True)
    
    return train_dataloader, test_dataloader

def get_unpaired_data(config):
    batch_size=config["batch_size"]
    torchtext.utils.download_from_url("https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip")
    torchtext.utils.extract_archive("./.data/horse2zebra.zip","data")
    
    train_horses = CustomUnpairedDataset(img_dir="data/horse2zebra", data="trainA", transform=True)
    train_zebras = CustomUnpairedDataset(img_dir="data/horse2zebra", data="trainB", transform=True)

    test_horses = CustomUnpairedDataset(img_dir="data/horse2zebra", data="testA", transform=False)
    test_zebras = CustomUnpairedDataset(img_dir="data/horse2zebra", data="testB", transform=False)

    horse_train_ds = DataLoader(train_horses, batch_size=batch_size, shuffle=True)
    zebra_train_ds = DataLoader(train_zebras, batch_size=batch_size, shuffle=True)
    
    horse_test_ds = DataLoader(test_horses, batch_size=batch_size, shuffle=True)
    zebra_test_ds = DataLoader(test_zebras, batch_size=batch_size, shuffle=True)
    
    return (horse_train_ds,zebra_train_ds),(horse_test_ds,zebra_test_ds)
    