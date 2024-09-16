'''
Modified from: https://github.com/Linear95/CLUB
'''

import torch
import torch.nn as nn
import sys
sys.path.append("/home/laiyy/code/FedSen")
from utils.conf import get_device
import numpy as np
import pdb


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            mi_est() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''

    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())
        # self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
        #                                 nn.ReLU(),
        #                                 nn.Linear(hidden_size//2, y_dim))

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def mi_est(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.


class CLUBSample_reshape(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample_reshape, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):  # (batch, len_crop/2, 1), (batch, len_crop/2, 64)
        mu, logvar = self.get_mu_logvar(x_samples)  # to (batch, len_crop/2, 64)
        mu = mu.reshape(-1, mu.shape[-1])  # (bs*T, y_dim)
        logvar = logvar.reshape(-1, logvar.shape[-1])  # (bs*T, y_dim)
        y_samples = y_samples.reshape(-1, y_samples.shape[-1])  # (bs, T, y_dim) -> (bs*T, y_dim)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def mi_est(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        sample_size = mu.shape[0]
        random_index = torch.randperm(sample_size).long()
        y_shuffle = y_samples[random_index]
        mu = mu.reshape(-1, mu.shape[-1])  # (bs, y_dim) -> (bs, 1, y_dim) -> (bs, T, y_dim) -> (bs*T, y_dim)
        logvar = logvar.reshape(-1, logvar.shape[-1])
        y_samples = y_samples.reshape(-1, y_samples.shape[-1])  # (bs, T, y_dim) -> (bs*T, y_dim)
        y_shuffle = y_shuffle.reshape(-1, y_shuffle.shape[-1])  # (bs, T, y_dim) -> (bs*T, y_dim)

        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_shuffle) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.


class CLUBSample_group(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample_group, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood, (batch, 256)， (batch, len_crop/2, 64)
        mu, logvar = self.get_mu_logvar(x_samples)  # mu/logvar: (bs, y_dim), to (batch, z_dim（64）)
        mu = mu.unsqueeze(1).expand(-1, y_samples.shape[1], -1).reshape(-1, mu.shape[
            -1])  # (bs, y_dim) -> (bs, 1, y_dim) -> (bs, T(len_crop/2), y_dim) -> (bs * T, y_dim)
        logvar = logvar.unsqueeze(1).expand(-1, y_samples.shape[1], -1).reshape(-1, logvar.shape[-1])
        y_samples = y_samples.reshape(-1, y_samples.shape[-1])  # (bs, T, y_dim) -> (bs * T, y_dim)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0) / 2

    def mi_est(self, x_samples, y_samples):     # (batch, 256), (bz, len_crop/2, 64)
        # x_samples: (bs, x_dim); y_samples: (bs, T, y_dim)
        mu, logvar = self.get_mu_logvar(x_samples)      # (batch, z_dim（64）)

        sample_size = x_samples.shape[0]        # batch
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()       # 将0~256-1（包括0和n-1）随机打乱后获得的数字序列

        # log of conditional probability of positive sample pairs
        mu_exp1 = mu.unsqueeze(1).expand(-1, y_samples.shape[1], -1)  # (bs, y_dim) -> (bs, T, y_dim)
        # logvar_exp1 = logvar.unqueeze(1).expand(-1, y_samples.shape[1], -1).reshape(-1, logvar.shape[-1])
        positive = - ((mu_exp1 - y_samples) ** 2).mean(dim=1) / logvar.exp()  # mean along T
        negative = - ((mu_exp1 - y_samples[random_index]) ** 2).mean(dim=1) / logvar.exp()  # mean along T

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean() / 2


def get_mi_model(device_id, x1, x2, epoch, mi_model_name='CLUB'):
    device = get_device(device_id)
    if mi_model_name == 'CLUB':
        mi_model = CLUB(x_dim=512, y_dim=512, hidden_size=512).to(device)
    elif mi_model_name == 'VarUB':
        mi_model = VarUB(x_dim=512, y_dim=512, hidden_size=512).to(device)
    elif mi_model_name == 'MINE':
        mi_model = MINE(x_dim=512, y_dim=512, hidden_size=512).to(device)
    elif mi_model_name == 'MINE2':
        mi_model = MINE2(x_dim=512, y_dim=512).to(device)
    optimizer_mi_model = torch.optim.Adam(mi_model.parameters(), lr=5e-3)
    x1 = x1.detach()
    x2 = x2.detach()
    for i in range(epoch):
        optimizer_mi_model.zero_grad()
        lld_loss = mi_model.learning_loss(x1, x2)
        lld_loss.backward()
        optimizer_mi_model.step()
        # print(f'Epoch {i}, lld_loss: {lld_loss}')
    return mi_model


def mi_estimate(C, D, order=1):
    z, y = D, C
    m, d = z.size()
    z, y = z.contiguous(), y.contiguous()
    A = torch.cdist(z, z, p=2)
    B = torch.cdist(y, y, p=2)
    A_row_sum, A_col_sum = A.sum(dim=0, keepdim=True), A.sum(dim=1, keepdim=True)
    B_row_sum, B_col_sum = B.sum(dim=0, keepdim=True), B.sum(dim=1, keepdim=True)
    a = A - A_row_sum/(m-2) - A_col_sum/(m-2) + A.sum()/((m-1)*(m-2))
    b = B - B_row_sum/(m-2) - B_col_sum/(m-2) + B.sum()/((m-1)*(m-2))
    if order == 1:
        AB, AA, BB = (a*b).sum()/(m*(m-3)), (a*a).sum()/(m*(m-3)), (b*b).sum()/(m*(m-3))
        mi = AB**0.5/(AA**0.5 * BB**0.5)**0.5
    else:
        # a, b = a.view(m*m, 1), b.view(m*m, 1)
        # c1, c2 = renyi_min(a, b)**0.5, renyi_min(b, a)**0.5
        # mi = c1 if c1>=c2 else c2
        mi = 0.0
    return mi


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 225 * 225, 512),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class VarUB(nn.Module):  #    variational upper bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(VarUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
            
    def forward(self, x_samples, y_samples): #[nsample, 1]
        mu, logvar = self.get_mu_logvar(x_samples)
        return 1./2.*(mu**2 + logvar.exp() - 1. - logvar).mean()
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.T_func(torch.cat([x_samples,y_shuffle], dim = -1))

        lower_bound = T0.mean() - torch.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


class MINE2(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=10):
        super(MINE2, self).__init__()
        self.layers = nn.Sequential(nn.Linear(2 * x_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
 
    def learning_loss(self, x, y):
        batch_size = x.size(0)
        tiled_x = torch.cat([x, x, ], dim=0)
        idx = torch.randperm(batch_size)
 
        shuffled_y = y[idx]
        concat_y = torch.cat([y, shuffled_y], dim=0)
        inputs = torch.cat([tiled_x, concat_y], dim=1)
        logits = self.layers(inputs)
 
        pred_xy = logits[:batch_size]
        pred_x_y = logits[batch_size:]
        loss = - np.log2(np.exp(1)) * (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))))
        # compute loss, you'd better scale exp to bit
        return loss
    
    def forward(self, x, y):
        return -self.learning_loss(x, y)


if __name__ == '__main__':
    x1 = torch.randn(64, 3, 225, 225).to('cuda:1')
    # x2 = torch.randn(64, 3, 225, 225).to('cuda:1')
    x2 = x1.clone().to('cuda:1')
    model = EmbeddingNet().to('cuda:1')
    z1 = model(x1)
    z2 = model(x2)
    mi_model = get_mi_model(1, z1, z2, 10, mi_model_name='CLUB')
    mi_loss = mi_model(z1, z2)
    print(mi_loss.item())