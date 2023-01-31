import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features), 
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, input_shape= (3,256,256), num_residual_block=9):
        super(ResnetGenerator, self).__init__()
        channels = input_shape[0]
        out_features = 64
        model = [nn.ReflectionPad2d(channels),nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),nn.ReLU(inplace=True)]
        in_features = out_features
        
        for _ in range(2):
            out_features *= 2
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),nn.ReLU(inplace=True)]
            in_features = out_features
        
        for _ in range(num_residual_block):
            model += [ResidualBlock(out_features)]
        
        for _ in range(2):
            out_features //= 2
            model += [nn.Upsample(scale_factor=2),nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.ReLU(inplace=True)]
            in_features = out_features
            
        model += [nn.ReflectionPad2d(channels),nn.Conv2d(out_features, channels, 7),nn.Tanh()]
        self.model = nn.Sequential(*model) 
        
    def forward(self, x):
        return self.model(x)