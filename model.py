import torch.nn as nn
import torch

class ConvVAE(nn.Module):
    def __init__(self, in_channels, out_channels, latent_size):
        super(ConvVAE, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1), # output 150x150 -> 150x150
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2), # output 150x150 -> 152x152
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 152 -> 76
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2), # output 76x76 -> 78x78
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 78 -> 39
        )
        
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2), # output 39x39 -> 41x41
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 41 -> 20
        )
        
        self.encoder5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2), # output 20x20 -> 22x22
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 22 -> 11
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # same: 11 -> 11
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # same: 11 -> 11
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.Upsample(size=(20,20),mode='nearest'), # output 11x11 -> 20x20
            nn.Conv2d(512,256, kernel_size=3, stride=1, padding=1), # same 20 -> 20
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.01,inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Upsample(size=(39,39),mode='nearest'), # output 20x20 -> 39x39
            nn.Conv2d(512,128, kernel_size=3, stride=1, padding=1), # same 39 -> 39
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.01,inplace=True)
        )
        
        self.decoder3 = nn.Sequential(
            nn.Upsample(size=(76,76),mode='nearest'), # output 39x39 -> 76x76
            nn.Conv2d(256,64, kernel_size=3, stride=1, padding=1), # same 76 -> 76
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.01,inplace=True)
        )
        self.decoder4 = nn.Sequential(
            nn.Upsample(size=(150,150),mode='nearest'), # output 76x76 -> 150x150
            nn.Conv2d(128,32, kernel_size=3, stride=1, padding=1), # same 150 -> 150
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(negative_slope=0.01,inplace=True)
        )
        
        self.decoder5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # same: 150 -> 150
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # same: 150 -> 150
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # same: 150 -> 150
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1), # same: 150 -> 150
        )
        
        ### Fully connected layers for mean and logvar ###
        self.mean = nn.Sequential(
            nn.Linear(512*11*11, latent_size),
            nn.BatchNorm1d(latent_size),  # Batch Normalization
            nn.ReLU(inplace=True),
            nn.Linear(latent_size, 512 * 11 * 11),
            nn.BatchNorm1d(512 * 11 * 11),  # Batch Normalization
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (512, 11, 11))  # Reshape to match the shape after encoder
        )
        self.logvar = nn.Sequential(
            nn.Linear(512*11*11, latent_size),
            nn.BatchNorm1d(latent_size),  # Batch Normalization
            nn.ReLU(inplace=True),
            nn.Linear(latent_size, 512*11*11),
            nn.BatchNorm1d(512*11*11),  # Batch Normalization
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (512, 11, 11))  # Reshape to match the shape after encoder
        )
        
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        return x1, x2, x3, x4, x5

    def decode(self, x1, x2, x3, x4, x5):
        z = self.decoder1(x5)
        z = self.decoder2(torch.cat((z,x4), dim=1))
        z = self.decoder3(torch.cat((z,x3), dim=1))
        z = self.decoder4(torch.cat((z,x2), dim=1))
        z = self.decoder5(torch.cat((z,x1), dim=1))
        return z

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encode(x)
        
        mean = self.mean(x5.reshape(x5.shape[0],-1))
        logvar = self.logvar(x5.reshape(x5.shape[0],-1))
        x5 = self.reparameterize(mean,logvar)
        
        x_recon = self.decode(x1, x2, x3, x4, x5)
        return x_recon, mean, logvar