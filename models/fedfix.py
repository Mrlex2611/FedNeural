import pdb
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from utils.args import *
from utils.util import dot_loss
from models.utils.federated_model import FederatedModel
from datasets.pacs import FedLeaPACS
from datasets.utils.federated_dataset import PseudoLabeledDataset
from backbone.ResNet import Classifier, ETF_Classifier
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

class FedFix(FederatedModel):
    NAME = 'fedfix'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list,args, transform):
        super(FedFix, self).__init__(nets_list,args,transform)
        self.global_proto = {}
        self.unlabel_loader_truth = {}
        self.dataset = FedLeaPACS
        self.epoch = 0

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _,net in enumerate(self.nets_list):
            net.load_state_dict(global_w)
        self.classifier = ETF_Classifier(feat_in=512, num_classes=self.dataset.N_CLASS).to(self.device)
        self.classifier.ori_M = self.classifier.ori_M.to(self.device)

    def loc_update_label(self,priloader_list,label_clients):
        # total_clients = list(range(self.args.parti_num))
        # online_clients = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        online_clients = label_clients
        self.online_clients = online_clients

        cur_M = self.classifier.ori_M
        for i in online_clients:
            self._train_net(i,self.nets_list[i], priloader_list[i], cur_M)
        self.aggregate_nets(None)
        self.epoch += 1

        return  None
    
    def loc_update_all(self,priloader_list,label_clients,unlabel_clients):
        # total_clients = list(range(self.args.parti_num))
        # online_clients = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        online_clients = label_clients + unlabel_clients
        self.online_clients = online_clients
        if self.epoch == self.args.pritrain_epoch:  # the first round of unlabeled training
            print('unlabel training start...')
            for i in unlabel_clients:
                self.unlabel_loader_truth[i] = copy.deepcopy(priloader_list[i])

        cur_M = self.classifier.ori_M
        for i in label_clients:
            self._train_net(i,self.nets_list[i], priloader_list[i], cur_M)
        for i in unlabel_clients:
            self._assign_pseudo_labels(priloader_list, i, cur_M)
            self._train_net(i,self.nets_list[i], priloader_list[i], cur_M)
        self.aggregate_nets(None)
        self.epoch += 1

        return  None
    
    def _assign_pseudo_labels(self,priloader_list,unlabel_client,cur_M):
        global_net = self.global_net
        valid_images = []
        valid_labels = []
        threshold = self.args.pseudo_label_threshold
        with torch.no_grad():
            total, correct = 0, 0
            for batch in self.unlabel_loader_truth[unlabel_client]:
                images, labels = batch[0], batch[1]
                images, labels = images.to(self.device), labels.to(self.device)
                feat = global_net(images)
                feat = self.classifier(feat)
                output = torch.matmul(feat, cur_M)
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
                
            top1acc = round(100 * correct / total, 2)
            print(f'Client {unlabel_client} total samples number: {len(self.unlabel_loader_truth[unlabel_client].dataset)}')
            print(f'Pseudo samples number: {total}, pseudo label assign accuracy: {top1acc}')

        if valid_labels:
            valid_images = torch.cat(valid_images)
            valid_labels = torch.cat(valid_labels)
        pseudo_labeled_dataset = PseudoLabeledDataset(valid_images, valid_labels)
        pseudo_labeled_dataloader = DataLoader(pseudo_labeled_dataset, batch_size=self.dataset.args.local_batch_size, shuffle=True)
        priloader_list[unlabel_client] = pseudo_labeled_dataloader


    def _train_net(self,index,net,train_loader,cur_M):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        # criterion = 'reg_dot_loss'
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                feat = net(images)
                feat = self.classifier(feat)
                output = torch.matmul(feat, cur_M)
                loss = criterion(output, labels)
                # with torch.no_grad():
                #     feat_nograd = feat.detach()
                #     H_length = torch.clamp(torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)), 1e-8)
                # loss = dot_loss(feat, labels, cur_M, self.classifier, criterion, H_length)

                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index,loss)
                optimizer.step()
        

        # calculate prototype
        net.eval()
        # 存储每个类别的表征和计数器
        class_representations = defaultdict(lambda: {'sum': 0, 'count': 0})
        for data, labels in train_loader:
            data = data.to(self.device)
            labels = labels.to(self.device)
            representations = net(data)

            # 累加每个类别的表征和计数
            for label, representation in zip(labels, representations):
                if label.item() not in class_representations:
                    class_representations[label.item()]['sum'] = representation
                    class_representations[label.item()]['count'] = 1
                else:
                    class_representations[label.item()]['sum'] = class_representations[label.item()]['sum'].clone() + representation
                    class_representations[label.item()]['count'] = class_representations[label.item()]['count'] + 1

        # 计算每个类别的平均表征作为原型
        net.proto = {label: info['sum'] / info['count'] for label, info in class_representations.items()}


    def aggregate_nets(self, freq=None):
        global_net = self.global_net
        nets_list = self.nets_list

        online_clients = self.online_clients
        global_w = self.global_net.state_dict()

        if self.args.averaing == 'weight':
            online_clients_dataset = [self.trainloaders[online_clients_index].dataset for online_clients_index in online_clients]
            online_clients_len = [len(dataset) for dataset in online_clients_dataset]
            online_clients_all = np.sum(online_clients_len)
            freq = online_clients_len / online_clients_all
        else:
        # if freq == None:
            parti_num = len(online_clients)
            freq = [1 / parti_num for _ in range(parti_num)]
        
        # 可能只有部分客户端有某一个类的原型，因此要按照比例放大对应的加权权重
        adjust = {}
        for c in range(self.dataset.N_CLASS):
            c_len = 0
            for client in online_clients:
                net = nets_list[client]
                if c in net.proto.keys():
                    c_len += 1
            adjust[c] = 1.0 * c_len / len(online_clients)
            

        first = True
        updated_proto = {}
        for index, net_id in enumerate(online_clients):
            net = nets_list[net_id]
            net_para = net.state_dict()
            if first:
                first = False
                for key in net_para:
                    if key in global_w.keys():
                        global_w[key] = net_para[key].clone() * freq[index]
            else:
                for key in net_para:
                    if key in global_w.keys():
                        global_w[key] = global_w[key] + net_para[key].clone() * freq[index]

            # update prototype 
            for key in net.proto:
                if key in updated_proto.keys():
                    updated_proto[key] = updated_proto[key] + net.proto[key].clone() * freq[index] / adjust[key]
                else:
                    updated_proto[key] = net.proto[key].clone() * freq[index] / adjust[key]

        global_net.load_state_dict(global_w)

        for _, net in enumerate(nets_list):
            net_para = net.state_dict()
            net_para.update({k: v for k, v in global_w.items() if k in net_para})
            net.load_state_dict(net_para)
        self.global_proto = updated_proto


        # update classifier
        initial_alpha = 1.0
        decay_rate = 0.001
        decay_per_epoch = 20
        alpha = initial_alpha - (self.epoch // decay_per_epoch) * decay_rate
        feat_dim, num_classes = self.classifier.ori_M.shape
        new_M = torch.zeros_like(self.classifier.ori_M)
        # 对每个类别进行加权
        for i in range(num_classes):
            ori_vec = self.classifier.ori_M[:, i]
            # 获取 updated_proto 中类别 i 的更新原型，如果没有则用 ori_vec 代替
            proto_vec = updated_proto.get(i, ori_vec)
            combined_vec = alpha * ori_vec + (1-alpha) * proto_vec
            normalized_vec = combined_vec / torch.norm(combined_vec, p=2)
            new_M[:, i] = normalized_vec
        with torch.no_grad():  # 禁止追踪此块的梯度
            self.classifier.ori_M.copy_(new_M)