import pdb
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

class FedSR(FederatedModel):
    NAME = 'fedsr'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list,args, transform):
        super(FedSR, self).__init__(nets_list,args,transform)
        # num_classes = 7 if args.dataset == 'fl_pacs' else 10
        
        self.L2R_coeff = 1e-2
        self.CMI_coeff = 5e-4
        # self.optim.add_param_group({'params':[self.r_mu,self.r_sigma,self.C],'lr':self.lr,'momentum':0.9})

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _,net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self,priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            # pdb.set_trace()
            self._train_net(i,self.nets_list[i], priloader_list[i])
        self.aggregate_nets(None)

        return  None

    def _train_net(self,index,net,train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        # optimizer.add_param_group({'params':[net.r_mu,net.r_sigma,net.C],'lr':self.local_lr,'momentum':0.9, 'weight_decay':1e-5})
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                z, (z_mu,z_sigma) = net.featurize(images,return_dist=True)
                outputs = net.classifier(z)
                loss = criterion(outputs, labels)

                obj = loss
                regL2R = torch.zeros_like(obj)
                regCMI = torch.zeros_like(obj)
                # regNegEnt = torch.zeros_like(obj)

                if self.L2R_coeff != 0.0:
                    regL2R = z.norm(dim=1).mean()
                    obj = obj + self.L2R_coeff*regL2R
                if self.CMI_coeff != 0.0:
                    r_sigma_softplus = F.softplus(net.r_sigma)
                    r_mu = net.r_mu[labels]
                    r_sigma = r_sigma_softplus[labels]
                    z_mu_scaled = z_mu*net.C
                    z_sigma_scaled = z_sigma*net.C
                    regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + \
                            (z_sigma_scaled**2+(z_mu_scaled-r_mu)**2)/(2*r_sigma**2) - 0.5
                    regCMI = regCMI.sum(1).mean()
                    obj = obj + self.CMI_coeff*regCMI

                # z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
                # mix_coeff = dist.categorical.Categorical(images.new_ones(images.shape[0]))
                # mixture = dist.mixture_same_family.MixtureSameFamily(mix_coeff,z_dist)
                # log_prob = mixture.log_prob(z)
                # regNegEnt = log_prob.mean()


                optimizer.zero_grad()
                obj.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f, loss_L2R = %0.3f, loss_MI = %0.3f" % (index,loss,regL2R,regCMI)
                optimizer.step()

