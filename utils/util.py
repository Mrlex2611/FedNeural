import os
import numpy as np
import random
from collections import defaultdict

import torch


def create_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def save_networks(model, communication_idx):
    nets_list = model.nets_list
    model_name = model.NAME

    checkpoint_path = model.checkpoint_path
    model_path = os.path.join(checkpoint_path, model_name)
    model_para_path = os.path.join(model_path, 'para')
    create_if_not_exists(model_para_path)
    for net_idx, network in enumerate(nets_list):
        each_network_path = os.path.join(model_para_path, str(communication_idx) + '_' + str(net_idx) + '.ckpt')
        torch.save(network.state_dict(), each_network_path)


def save_protos(model, communication_idx):
    model_name = model.NAME

    checkpoint_path = model.checkpoint_path
    model_path = os.path.join(checkpoint_path, model_name)
    model_para_path = os.path.join(model_path, 'protos')
    create_if_not_exists(model_para_path)

    for i in range(len(model.global_protos_all)):
        label = i
        protos = torch.cat(model.global_protos_all[i], dim=0).cpu().numpy()
        save_path = os.path.join(model_para_path, str(communication_idx) + '_' + str(label) + '.npy')
        np.save(save_path, protos)

def dot_loss(output, label, cur_M, classifier, criterion, H_length, reg_lam=0):
    target = cur_M[:, label].T ## B, d  output: B, d
    if criterion == 'dot_loss':
        loss = - torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1).mean()
    elif criterion == 'reg_dot_loss':
        dot = torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1) #+ classifier.module.bias[label].view(-1)

        with torch.no_grad():
            M_length = torch.sqrt(torch.sum(target ** 2, dim=1, keepdims=False))
        loss = (1/2) * torch.mean(((dot-(M_length * H_length)) ** 2) / H_length)

        if reg_lam > 0:
            reg_Eh_l2 = torch.mean(torch.sqrt(torch.sum(output ** 2, dim=1, keepdims=True)))
            loss = loss + reg_Eh_l2*reg_lam

    return loss

def sample_unlabel_clients(selected_domain_list, total_clients, sample_ratio=0.9):
    if selected_domain_list is None:
        sample_count = int(len(total_clients) * sample_ratio) if total_clients else 0
        sampled_indices = list(random.sample(total_clients, sample_count))
    else:
        domain_indices = defaultdict(list)
        for idx, domain in enumerate(selected_domain_list):
            domain_indices[domain].append(idx)
        
        # sample a certain ratio of idxs from each doamin
        sampled_indices = []
        for indices in domain_indices.values():
            sample_count = int(len(indices) * sample_ratio) if indices else 0
            sampled_indices.extend(random.sample(indices, sample_count))
    
    return sampled_indices