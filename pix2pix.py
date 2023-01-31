from unet_generator import UnetGenerator
from discriminator import Discriminator
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

class pix2pix:
    def __init__(self,config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gen = UnetGenerator().to(self.device)
        self.dis = Discriminator().to(self.device)
        self.epochs = config["epochs"]
        self.lrs = config["lrs"]
        self.lambdaa = config["lambda"]
        self.gen_optim = torch.optim.Adam(self.gen.parameters(), lr=self.lrs[0],betas=(0.5,0.999))
        self.dis_optim = torch.optim.Adam(self.dis.parameters(), lr=self.lrs[1],betas=(0.5,0.999))
    
    def generate_images(self,test_input, tar,idx):
        test_input=test_input.float()
        prediction = self.gen(test_input.to(self.device)).detach().cpu()
        plt.figure(figsize=(15, 15))
        
        display_list = [test_input[idx], tar[idx], prediction[idx]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow((display_list[i] * 0.5 + 0.5).permute(1, 2, 0))
            plt.axis('off')
        plt.savefig(f'out{idx}')
    
    def train(self, train_data, test_data):
        for epoch in range(self.epochs):
            test_inp, test_tar = next(iter(test_data))
            running_loss = np.array([0,0,0,0],dtype=float)
            for batch_idx, (inp,tar) in enumerate(train_data):
                b_size = inp.shape[0]
                real_class = torch.ones(b_size,1,30,30).to(self.device)
                fake_class = torch.zeros(b_size,1,30,30).to(self.device)

                #Train D
                self.dis.zero_grad()
                real_patch = self.dis(inp,tar)
                real_gan_loss= nn.BCEWithLogitsLoss()(real_patch,real_class)

                fake=self.gen(inp)

                fake_patch = self.dis(inp,fake.detach())
                fake_gan_loss=nn.BCEWithLogitsLoss()(fake_patch,fake_class)

                D_loss = real_gan_loss + fake_gan_loss
                D_loss.backward()
                self.dis_optim.step()

                #Train G
                self.gen.zero_grad()
                fake_patch = self.dis(inp,fake)
                fake_gan_loss=nn.BCEWithLogitsLoss()(fake_patch,real_class)

                L1_loss = nn.L1Loss()(fake,tar)
                G_loss = fake_gan_loss + self.lambdaa*L1_loss
                G_loss.backward()

                self.gen_optim.step()

                x= np.array([G_loss.item(),fake_gan_loss.item(),L1_loss.item(),D_loss.item()],dtype=float)
                running_loss += x
                if (batch_idx+1)%4==0:
                    print("Step:",batch_idx+1,"Gen loss:",round(G_loss.item(),2),"Gen GAN loss:",round(fake_gan_loss.item(),2),"Gen L1:",round(L1_loss.item(),2),"Dis loss:",round(D_loss.item(),2))

            n=(batch_idx+1)
            running_loss=np.around(running_loss/n,decimals=2)
            print("Epoch:",epoch+1,"Gen loss:",running_loss[0],"Gen GAN loss:",running_loss[1],"Gen L1:",running_loss[2],"Dis loss:",running_loss[3])
            if (epoch+1)%5==0: 
                self.generate_images(test_inp,test_tar,0)
    
    def save_models(self, gen_ckpt, dis_ckpt):
        torch.save(self.gen.state_dict(),gen_ckpt)
        torch.save(self.dis.state_dict(),dis_ckpt)
        print("Models saved")
    
    def load_models(self, gen_ckpt, dis_ckpt):
        self.gen.load_state_dict(torch.load(gen_ckpt,map_location=torch.device('cpu')))
        self.dis.load_state_dict(torch.load(dis_ckpt,map_location=torch.device('cpu') ))
        print("Models loaded")