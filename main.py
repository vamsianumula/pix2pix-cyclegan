from pix2pix import pix2pix
from cyclegan import cycleGAN
from dataloader import get_paired_data, get_unpaired_data
import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

pix2pix_config = {
    "lrs":(2e-4,2e-4),
    "batch_size": 4,
    "epochs": 300,
    "lambda": 100 
}

p2p = pix2pix(pix2pix_config)
p2p.load_models("checkpoints/PIX2PIX_GEN.ckpt","checkpoints/PIX2PIX_DIS.ckpt")
train_data, test_data = get_paired_data(pix2pix_config)
# p2p.train(train_data,test_data)
test_inp, test_tar = next(iter(test_data))
for i in range(len(test_inp)):
    p2p.generate_images(test_inp,test_tar,i)

# cyclegan_config = {
#     "lrs":(1e-3, 2e-4,2e-4),
#     "batch_size": 1,
#     "epochs": 300,
#     "lambda": 10 
# }

# train_data, test_data = get_unpaired_data(cyclegan_config)
# cg = cycleGAN(cyclegan_config)
# # cg.train(train_data[0],train_data[1])
# cg.load_models(20,'checkpoints')
# test_horses = next(iter(test_data[0]))
# test_zebras = next(iter(test_data[1]))
# cg.generate_images(test_horses[0],test_zebras[0])

