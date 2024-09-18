import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions


class CNN_encoder(nn.Module):

    def __init__(self, dim_out=512):
        super(CNN_encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 7)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(128)
        self.act = nn.ReLU()
        self.fc = nn.Linear(128*23*23, dim_out)
        
        self.encoder = nn.Sequential(
            self.conv1,
            self.act,
            self.conv2,
            self.act,
            self.conv3,
            self.act,
            self.conv4
        )
    
    def forward(self, x):
        x = self.encoder(x)
        y = self.fc(x.view(-1, 128*23*23))
        return y


class CNN_decoder(nn.Module):

    def __init__(self, dim_in=512*2):
        super(CNN_decoder, self).__init__()
        self.reconv1 = nn.ConvTranspose2d(128, 64, 7)
        self.reconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.reconv3 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1)
        self.reconv4 = nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(16)
        self.act = nn.ReLU()
        self.fc = nn.Linear(dim_in, 128*23*23)
        
        self.decoder = nn.Sequential(
            self.reconv1,
            self.act,
            self.reconv2,
            self.act,
            self.reconv3,
            self.act,
            self.reconv4,
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.fc(x)
        y = self.decoder(x.view(-1, 128, 23, 23))
        return y


class CNN_encoder_simple(nn.Module):

    def __init__(self, dim_out=512):
        super(CNN_encoder_simple, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.act = nn.ReLU()
        self.fc = nn.Linear(32 * 8 * 8, dim_out)
        
        self.encoder = nn.Sequential(
            self.conv1,
            self.act,
            self.conv2,
            self.act
        )
    
    def forward(self, x):
        x = F.interpolate(x, size=(32, 32))  # 将输入图像压缩到32x32
        x = self.encoder(x)
        y = self.fc(x.view(-1, 32 * 8 * 8))
        return y


class CNN_decoder_simple(nn.Module):

    def __init__(self, dim_in=512*2):
        super(CNN_decoder_simple, self).__init__()
        self.reconv1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.reconv2 = nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1)
        self.act = nn.ReLU()
        self.fc = nn.Linear(dim_in, 32 * 8 * 8)
        
        self.decoder = nn.Sequential(
            self.reconv1,
            self.act,
            self.reconv2,
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.fc(x)
        y = self.decoder(x.view(-1, 32, 8, 8))
        return y


class AE(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 512, name: str = 'autoencoder'):
        super(AE, self).__init__()
        self.name=name
        self.proto = {}
        self.encoder = CNN_encoder_simple()
        self.semantic_branch = nn.Linear(hidden_size, hidden_size)
        self.context_branch = nn.Linear(hidden_size, hidden_size)
        self.cls = nn.Linear(hidden_size, num_classes)
    
    def semantic_feature(self, x):
        fea = self.encoder(x)
        fea = self.semantic_branch(fea)
        return fea
    
    def context_feature(self, x):
        fea = self.encoder(x)
        fea = self.context_branch(fea)
        return fea
    
    def classifier(self, x):
        x = self.encoder(x)
        out = self.cls(x)
        return out
    
    def context_classifier(self, x):
        z2 = self.encoder(x)
        z2 = self.context_branch(z2)
        out = self.cls(z2)
        return out

    def forward(self, x):
        z1 = self.encoder(x)
        z1 = self.semantic_branch(z1)
        out = self.cls(z1)
        return out


class AECLS(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 512, name: str = 'mycnn'):
        super(AECLS, self).__init__()
        self.name=name
        self.proto = {}
        self.encoder = CNN_encoder_simple()
        self.semantic_branch = nn.Linear(hidden_size, hidden_size)
        self.cls = nn.Linear(hidden_size, num_classes)
    
    def semantic_feature(self, x):
        fea = self.encoder(x)
        fea = self.semantic_branch(fea)
        return fea
    
    def features(self, x):
        fea = self.encoder(x)
        fea = self.semantic_branch(fea)
        return fea
    
    def classifier(self, fea):
        out = self.cls(fea)
        return out

    def forward(self, x):
        fea = self.encoder(x)
        fea = self.semantic_branch(fea)
        # out = self.cls(fea)
        return fea


class VAE(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 512, name: str = 'myvae'):
        super(VAE, self).__init__()
        self.name=name
        self.proto = {}
        if name == 'myvae':
            self.encoder = CNN_encoder_simple(dim_out=hidden_size)
            dim_in = hidden_size
        elif name == 'myvae_3':
            self.encoder = CNN_encoder(dim_out=hidden_size)
        self.z_dim = hidden_size
        self.mu_s = nn.Linear(hidden_size, hidden_size)
        self.log_sigma_s = nn.Linear(hidden_size, hidden_size)
        self.mu_d = nn.Linear(hidden_size, hidden_size)
        self.log_sigma_d = nn.Linear(hidden_size, hidden_size)
        self.mu_decoder = nn.Linear(2*hidden_size, 32*32*3)
    
    def featurize_s(self,x,num_samples=1,return_dist=False):
        x = self.encoder(x)
        mu = self.mu_s(x)
        log_sigma = self.log_sigma_s(x)
        sigma = torch.exp(log_sigma)
        # 采样，获取采样数据
        eps = torch.randn_like(sigma)  #eps: bs,latent_size
        z = mu + eps*sigma

        if return_dist:
            return z, mu, sigma
        else:
            return z

    def featurize_d(self,x,num_samples=1,return_dist=False):
        x = self.encoder(x)
        mu = self.mu_d(x)
        log_sigma = self.log_sigma_d(x)
        sigma = torch.exp(log_sigma)
        # 采样，获取采样数据
        eps = torch.randn_like(sigma)  #eps: bs,latent_size
        z = mu + eps*sigma
        
        if return_dist:
            return z, mu, sigma
        else:
            return z
    
    def semantic_feature(self, x):
        fea, mu_s, sigma_s = self.featurize_s(x, return_dist=True)
        return mu_s

    def forward(self, x):
        z1, mu_s, sigma_s = self.featurize_s(x, return_dist=True)
        z2, mu_d, sigma_d = self.featurize_d(x, return_dist=True)
        z3 = torch.cat((z1, z2), dim=1)
        x_ = self.mu_decoder(z3)
        x = F.interpolate(x, size=(32, 32))
        return x.reshape(x.size(0), -1), x_, mu_s, sigma_s, mu_d, sigma_d


class VAECLS(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 512, name: str = 'mycnn_vae'):
        super(VAECLS, self).__init__()
        self.name=name
        self.z_dim = hidden_size
        if name == 'mycnn_vae':
            self.encoder = CNN_encoder_simple(dim_out=hidden_size)
            dim_in = hidden_size
        elif name == 'mycnn_vae_3':
            self.encoder = CNN_encoder(dim_out=hidden_size)
        self.mu_s = nn.Linear(hidden_size, hidden_size)
        self.log_sigma_s = nn.Linear(hidden_size, hidden_size)
    
    def featurize(self,x,num_samples=1,return_dist=False):
        x = self.encoder(x)
        mu = self.mu_s(x)
        log_sigma = self.log_sigma_s(x)
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(sigma)  #eps: bs,latent_size
        z = mu + eps * sigma

        if return_dist:
            return z, mu, sigma
        else:
            return mu

    def forward(self, x):
        fea = self.featurize(x)
        return fea


def autoencoder(nclasses: int):
    return AE(nclasses)

def mycnn(nclasses: int):
    return AECLS(nclasses)

def mycnn_vae(nclasses: int):
    return VAECLS(nclasses)

def myvae(nclasses: int):
    return VAE(nclasses, name='myvae')

def myvae_cls(nclasses: int):
    return VAECLS(nclasses)