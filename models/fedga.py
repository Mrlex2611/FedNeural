import pdb
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

class FedGA(FederatedModel):
    NAME = 'fedga'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list,args, transform):
        super(FedGA, self).__init__(nets_list,args,transform)
        self.freq = None

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
        self.aggregate_nets()

        return  None

    def _train_net(self,index,net,train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.5, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index,loss)
                optimizer.step()

    def aggregate_nets(self):
        global_net = self.global_net
        nets_list = self.nets_list

        online_clients = self.online_clients
        global_w = self.global_net.state_dict()

        if self.args.averaing == 'weight':
            if self.freq is None:
                online_clients_dl = [self.trainloaders[online_clients_index] for online_clients_index in online_clients]
                online_clients_len = [dl.sampler.indices.size for dl in online_clients_dl]
                online_clients_all = np.sum(online_clients_len)
                self.freq = online_clients_len / online_clients_all
            else:
                #TODO 用GA策略更新self.freq
                online_clients_dl = [self.trainloaders[online_clients_index] for online_clients_index in online_clients]
                loss_before_avg, loss_after_avg = self.client_evaluate(self.trainloaders)
                weight_list = self.refine_weight_dict_by_GA(self.freq, loss_before_avg, loss_after_avg)
                self.freq = weight_list
        else:
        # if freq == None:
            parti_num = len(online_clients)
            self.freq = [1 / parti_num for _ in range(parti_num)]
        
        freq = self.freq

        first = True
        for index,net_id in enumerate(online_clients):
            net = nets_list[net_id]
            net_para = net.state_dict()
            # if net_id == 0:
            if first:
                first = False
                for key in net_para:
                    global_w[key] = net_para[key] * freq[index]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * freq[index]

        global_net.load_state_dict(global_w)

        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())
    
    def refine_weight_dict_by_GA(self, weight_list, site_before_results_dict, site_after_results_dict, step_size=0.1, fair_metric='loss'):
        if fair_metric == 'acc':
            signal = -1.0
        elif fair_metric == 'loss':
            signal = 1.0
        else:
            raise ValueError('fair_metric must be acc or loss')
        
        value_list = []
        for online_clients_index in self.online_clients:
            value_list.append(site_after_results_dict[online_clients_index] - site_before_results_dict[online_clients_index])
        
        value_list = np.array(value_list)
        
        
        step_size = 1./3. * step_size
        norm_gap_list = value_list / np.max(np.abs(value_list))
        
        for online_clients_index in self.online_clients:
            weight_list[online_clients_index] += signal * norm_gap_list[online_clients_index] * step_size

        weight_list = self.weight_clip(weight_list)
        
        return weight_list

    def weight_clip(self, weight_list):
        new_total_weight = 0.0
        for online_clients_index in self.online_clients:
            weight_list[online_clients_index] = np.clip(weight_list[online_clients_index], 0.0, 1.0)
            new_total_weight += weight_list[online_clients_index]
        
        for online_clients_index in self.online_clients:
            weight_list[online_clients_index] /= new_total_weight
        
        return weight_list
    
    def client_evaluate(self, test_dl: DataLoader):
        loss_before_avg = {}
        loss_after_avg = {}
        global_net = self.global_net
        global_net.eval()
        for j, dl in enumerate(test_dl):
            net = self.nets_list[j]
            net.eval()
            local_loss, global_loss = 0.0, 0.0
            total = 0
            criterion = nn.CrossEntropyLoss()
            for batch_idx, (images, labels) in enumerate(dl):
                with torch.no_grad():
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs_local = net(images)
                    output_global = global_net(images)
                    local_loss += criterion(outputs_local, labels).item()
                    global_loss += criterion(output_global, labels).item()
                    total += 1
            loss_before_avg[j] = local_loss / total
            loss_after_avg[j] = global_loss / total
        return loss_before_avg, loss_after_avg