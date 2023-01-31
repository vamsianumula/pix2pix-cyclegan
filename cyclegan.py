from resnet_generator import ResnetGenerator
from discriminator import DiscriminatorInstance
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import itertools

class cycleGAN:
    def __init__(self,config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.g = ResnetGenerator().to(self.device)
        self.f = ResnetGenerator().to(self.device)
        self.dx = DiscriminatorInstance().to(self.device)
        self.dy = DiscriminatorInstance().to(self.device)
        self.epochs = config["epochs"]
        self.lrs = config["lrs"]
        self.lambdaa = config["lambda"]
        self.g_optim = torch.optim.Adam(itertools.chain(self.g.parameters(),self.f.parameters()), lr=self.lrs[0],betas=(0.5,0.999))
        self.dx_optim = torch.optim.Adam(self.dx.parameters(), lr=self.lrs[1],betas=(0.5,0.999))
        self.dy_optim = torch.optim.Adam(self.dy.parameters(), lr=self.lrs[2],betas=(0.5,0.999))
    
    def generate_images(self,sample_horse, sample_zebra):
        to_zebra = self.g(sample_horse.unsqueeze(0).to(self.device)).detach().cpu()
        to_horse = self.f(sample_zebra.unsqueeze(0).to(self.device)).detach().cpu()
        plt.figure(figsize=(8, 8))

        imgs = [sample_horse, to_zebra, sample_zebra, to_horse]
        title = ['Horse', 'To Zebra', 'Zebra', 'To Horse']
    #     plt.imshow(imgs[i][0].permute(1, 2, 0) * 0.5 + 0.5)
        for i in range(len(imgs)):
            plt.subplot(2, 2, i+1)
            plt.title(title[i])
            if i % 2 == 0:
                plt.imshow(imgs[i].permute(1, 2, 0))
            else:
                plt.imshow(imgs[i][0].permute(1, 2, 0) * 0.5 + 0.5)
        plt.savefig(f'out.png')
    
    def train(self, train_horse, train_zebra):
        for epoch in range(self.epochs):
            running_loss = np.array([0,0,0,0,0],dtype=float)
            for batch_idx, (real_x,real_y) in enumerate(zip(train_horse,train_zebra)):
                self.g_optim.zero_grad()
                real_x = real_x.to(self.device)
                real_y = real_y.to(self.device)
                
                id_f_loss = nn.L1Loss()(self.f(real_x),real_x)
                id_g_loss= nn.L1Loss()(self.g(real_y),real_y)
                total_identity_loss = 0.5*(id_g_loss+ id_f_loss)
                
                fake_y = self.g(real_x)
                disc_fake_y = self.dy(fake_y)
                real = torch.ones(disc_fake_y.shape).to(self.device)
                fake = torch.zeros(disc_fake_y.shape).to(self.device)
                
                gen_g_loss = nn.MSELoss()(disc_fake_y, real)
                fake_x = self.f(real_y)
                disc_fake_x = self.dx(fake_x)
                gen_f_loss = nn.MSELoss()(disc_fake_x, real)
                total_gen_loss = 0.5*(gen_g_loss+ gen_f_loss)
                
            
                cycled_x = self.f(fake_y)
                cycle_f_loss = nn.L1Loss()(cycled_x, real_x)
                cycled_y = self.g(fake_x)
                cycle_g_loss = nn.L1Loss()(cycled_y, real_y)
                total_cycle_loss = 0.5*(cycle_g_loss+ cycle_f_loss)
                
                total_loss = total_gen_loss+0.5*self.lambdaa*total_identity_loss+ self.lambdaa*total_cycle_loss
                total_loss.backward()
                self.g_optim.step()
                
                
                self.dx_optim.zero_grad()
                disc_real_x = self.dx(real_x)
                disc_x_loss = 0.5*nn.MSELoss()(disc_real_x,real)
                disc_fake_x = self.dx(fake_x.detach())
                disc_x_loss += 0.5*nn.MSELoss()(disc_fake_x,fake)
                disc_x_loss.backward()
                self.dx_optim.step()
                
                self.dy_optim.zero_grad()
                disc_real_y = self.dy(real_y)
                disc_y_loss = 0.5*nn.MSELoss()(disc_real_y,real)
                disc_fake_y = self.dy(fake_y.detach())
                disc_y_loss += 0.5*nn.MSELoss()(disc_fake_y,fake)
                disc_y_loss.backward()
                self.dy_optim.step()
                x= np.array([total_gen_loss.item(),total_identity_loss.item(),total_cycle_loss.item(),disc_x_loss.item(),disc_y_loss.item()],dtype=float)
                # if batch_idx%4==0:
                print("Step:",batch_idx,"GAN:",round(x[0],2),"Id:",round(x[1],2),"Cyc:",round(x[2],2),"DX:",round(x[3],2),"DY:",round(x[4],2))
                running_loss += x
                
            n=(batch_idx+1)
            running_loss=np.around(running_loss/n,decimals=2)
            print("Epoch:",epoch+1,"GAN:",running_loss[0],"Id:",running_loss[1],"Cyc:",running_loss[2],"DX:",running_loss[3],"DY:",running_loss[4])
            if epoch%10==0:
                self.save_models(epoch) 
                self.generate_images(train_zebra[3],train_horse[10])
                
    def save_models(self, epoch):
        torch.save(self.g.state_dict(), f'g{epoch}.ckpt')
        torch.save(self.f.state_dict(), f'f{epoch}.ckpt')
        torch.save(self.dx.state_dict(), f'dx{epoch}.ckpt')
        torch.save(self.dy.state_dict(), f'dy{epoch}.ckpt')
        print("Models saved")
    
    def load_models(self, epoch, path='.'):
        self.g.load_state_dict(torch.load(f'{path}/g{epoch}.ckpt',map_location=torch.device('cpu')))
        self.f.load_state_dict(torch.load(f'{path}/f{epoch}.ckpt',map_location=torch.device('cpu')))
        self.dx.load_state_dict(torch.load(f'{path}/dx{epoch}.ckpt',map_location=torch.device('cpu') ))
        self.dy.load_state_dict(torch.load(f'{path}/dy{epoch}.ckpt',map_location=torch.device('cpu') ))
        print("Models loaded")