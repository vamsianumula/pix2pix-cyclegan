from pix2pix import pix2pix
from dataloader import get_paired_data
import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

config = {
    "lrs":(2e-4,2e-4),
    "batch_size": 4,
    "epochs": 300,
    "lambda": 100 
}

p2p = pix2pix(config)
p2p.load_models("checkpoints/PIX2PIX_GEN.ckpt","checkpoints/PIX2PIX_DIS.ckpt")
train_data, test_data = get_paired_data(config)
# p2p.train(train_data,test_data)
test_inp, test_tar = next(iter(test_data))
for i in range(len(test_inp)):
    p2p.generate_images(test_inp,test_tar,i)

