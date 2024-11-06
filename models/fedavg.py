import pdb
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel
from backbone.ResNet import Classifier, ETF_Classifier
from datasets.utils.federated_dataset import PseudoLabeledDataset
from datasets.pacs import FedLeaPACS
from datasets.cifar10 import FedLeaCifar10
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

class FedAvG(FederatedModel):
    NAME = 'fedavg'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list,args, transform):
        super(FedAvG, self).__init__(nets_list,args,transform)
        self.unlabel_loader_truth = {}
        self.dataset = FedLeaCifar10
        self.epoch = 0

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
    
    def loc_update_label(self,priloader_list,label_clients):
        # total_clients = list(range(self.args.parti_num))
        # online_clients = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        online_clients = label_clients
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i,self.nets_list[i], priloader_list[i])
        self.aggregate_nets(None)
        self.epoch += 1

        return  None
    
    def loc_update_all(self,priloader_list,label_clients,unlabel_clients):
        # total_clients = list(range(self.args.parti_num))
        # online_clients = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        online_clients = label_clients[:]
        if self.epoch == self.args.pritrain_epoch:  # the first round of unlabeled training
            print('unlabel training start...')
            for i in unlabel_clients:
                self.unlabel_loader_truth[i] = copy.deepcopy(priloader_list[i])

        for i in label_clients:
            self._train_net(i,self.nets_list[i], priloader_list[i])
        for i in unlabel_clients:
            if self._assign_pseudo_labels(priloader_list, i):
                online_clients.append(i)
                self._train_net(i,self.nets_list[i], priloader_list[i])
        self.online_clients = online_clients[:]
        self.aggregate_nets(None)
        self.epoch += 1

        return  None
    
    def _assign_pseudo_labels(self,priloader_list,unlabel_client):
        global_net = self.global_net.to(self.device)
        valid_images = []
        valid_labels = []
        threshold = self.args.pseudo_label_threshold
        total, correct = 0, 0
        with torch.no_grad():
            for batch in self.unlabel_loader_truth[unlabel_client]:
                images, labels = batch[0], batch[1]
                images, labels = images.to(self.device), labels.to(self.device)
                output = global_net(images)
                probabilities = F.softmax(output, dim=1)  # convert to probabilities
                max_probs, predicted_labels = torch.max(probabilities, dim=1)

                # keep confident samples
                mask = max_probs > threshold
                pseudo_images = images[mask]
                pseudo_labels = predicted_labels[mask]
                truth_labels = labels[mask]
                # calculate pseudo label assigning accuracy
                correct += (pseudo_labels == truth_labels).sum().item()
                total += len(pseudo_labels)


                if len(pseudo_images) > 0:
                    valid_images.append(pseudo_images)
                    valid_labels.append(pseudo_labels)
        
        if total <= 1:
            print(f'Client {unlabel_client} total samples number: {len(self.unlabel_loader_truth[unlabel_client].sampler)}')
            print(f'Pseudo samples number: {total}')
            return False    # no more than one pseudo sample meet the threshold requirement
        else:
            top1acc = round(100 * correct / total, 2)
            print(f'Client {unlabel_client} total samples number: {len(self.unlabel_loader_truth[unlabel_client].sampler)}')
            print(f'Pseudo samples number: {total}, pseudo label assign accuracy: {top1acc}')

            if valid_labels:
                valid_images = torch.cat(valid_images)
                valid_labels = torch.cat(valid_labels)
            pseudo_labeled_dataset = PseudoLabeledDataset(valid_images, valid_labels)
            pseudo_labeled_dataloader = DataLoader(pseudo_labeled_dataset, batch_size=self.args.local_batch_size, shuffle=True)
            priloader_list[unlabel_client] = pseudo_labeled_dataloader
            return True

    def _train_net(self,index,net,train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
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

