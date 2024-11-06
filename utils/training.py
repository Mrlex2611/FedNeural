import pdb
import torch
from argparse import Namespace
from models.utils.federated_model import FederatedModel
from datasets.utils.federated_dataset import FederatedDataset
from utils.util import sample_unlabel_clients
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
from utils.logger import CsvWriter
from collections import Counter
import random

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os


def global_evaluate(model: FederatedModel, test_dl: DataLoader, setting: str, name: str) -> Tuple[list, list]:
    accs = []
    net = model.global_net.to(model.device)
    status = net.training
    net.eval()
    for j, dl in enumerate(test_dl):
        correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(dl):
            with torch.no_grad():
                images, labels = images.to(model.device), labels.to(model.device)
                outputs = net(images)
                if model.NAME in ['fedfix']:
                    outputs = model.classifier(outputs)
                    outputs = torch.matmul(outputs, model.classifier.ori_M)

                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()
                top5 += (labels == max5).sum().item()
                total += labels.size(0)
        top1acc = round(100 * top1 / total, 2)
        top5acc = round(100 * top5 / total, 2)
        # if name in ['fl_digits','fl_officecaltech']:
        accs.append(top1acc)
        # elif name in ['fl_office31','fl_officehome']:
        #     accs.append(top5acc)
    net.train(status)
    return accs


def tsne_visual(model: FederatedModel, test_dl: DataLoader, setting: str, name: str) -> Tuple[list, list]:
    features_list = []  # 用于收集特征
    labels_list = []   # 用于收集标签
    net = model.global_net
    status = net.training
    net.eval()
    for j, dl in enumerate(test_dl):
        correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(dl):
            # images, labels = images.to(model.device), labels.to(model.device)
            with torch.no_grad():
                outputs = net(images)
                features = net.semantic_feature(images)  # 收集特征
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
                labels = labels.view(-1, 1)
    net.train(status)

    # 对特征进行t-SNE降维
    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    tsne = TSNE(n_components=2, random_state=0)
    reduced_features = tsne.fit_transform(features_array)

    # 可视化，为每个类别分别绘制并创建图例
    plt.figure()
    unique_labels = np.unique(labels_array)
    for label in unique_labels:
        indices = labels_array == label
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=f'Class {label}', s=10)
    plt.legend()
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.title('t-SNE Visualization of Model Features')
    save_path = f'exp_results/office/t-sne/{model.NAME}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path+f'{model.NAME}_global_tsne_200epoch.png')


def tsne_visual_local(model: FederatedModel, test_dl: DataLoader, parti_num: int, setting: str, name: str) -> Tuple[list, list]:
    for i in range(parti_num):
        features_list = []  # 用于收集特征
        labels_list = []   # 用于收集标签
        net = model.nets_list[i]
        status = net.training
        net.eval()
        for j, dl in enumerate(test_dl):
            correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
            for batch_idx, (images, labels) in enumerate(dl):
                images, labels = images.to(model.device), labels.to(model.device)
                with torch.no_grad():
                    outputs = net(images)
                    features = net.semantic_feature(images)  # 收集特征
                    features_list.append(features.cpu().numpy())
                    labels_list.append(labels.cpu().numpy())
                    labels = labels.view(-1, 1)
        net.train(status)

        # 对特征进行t-SNE降维
        features_array = np.concatenate(features_list, axis=0)
        labels_array = np.concatenate(labels_list, axis=0)
        tsne = TSNE(n_components=2, random_state=0)
        reduced_features = tsne.fit_transform(features_array)

        # 可视化，为每个类别分别绘制并创建图例
        plt.figure()
        unique_labels = np.unique(labels_array)
        for label in unique_labels:
            indices = labels_array == label
            plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=f'Class {label}', s=10)
        plt.legend()
        plt.xlabel('t-SNE feature 1')
        plt.ylabel('t-SNE feature 2')
        plt.title(f'Client {i} t-SNE Visualization of Model Features')
        save_path = f'exp_results/office/t-sne/{model.NAME}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path+f'{model.NAME}_client{i}_tsne_200epoch.png')


def train(model: FederatedModel, private_dataset: FederatedDataset,
          args: Namespace) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)

    model.N_CLASS = private_dataset.N_CLASS
    if private_dataset.NAME not in ['fl_cifar10']:      # multi-domain dataset
        domains_list = private_dataset.DOMAINS_LIST
        domains_len = len(domains_list)

        if args.rand_dataset:
            max_num = 10
            is_ok = False

            while not is_ok:
                if model.args.dataset == 'fl_officecaltech':
                    selected_domain_list = np.random.choice(domains_list, size=args.parti_num - domains_len, replace=True, p=None)
                    selected_domain_list = list(selected_domain_list) + domains_list
                elif model.args.dataset == 'fl_digits':
                    selected_domain_list = np.random.choice(domains_list, size=args.parti_num, replace=True, p=None)

                result = dict(Counter(selected_domain_list))

                for k in result:
                    if result[k] > max_num:
                        is_ok = False
                        break
                else:
                    is_ok = True

        else:
            # selected_domain_dict = {'mnist': 6, 'usps': 4, 'svhn': 3, 'syn': 7}  # base
            # selected_domain_dict = {'mnist': 1, 'usps': 1, 'svhn': 9, 'syn': 9}  # 20

            # selected_domain_dict = {'mnist': 3, 'usps': 2, 'svhn': 1, 'syn': 4}  # 10
            # selected_domain_dict = {'caltech': 3, 'amazon': 2, 'webcam': 1, 'dslr': 4}  # for office caltech
            # selected_domain_dict = {'photo': 1, 'art_painting': 2, 'cartoon': 3, 'sketch': 4}
            # selected_domain_dict = {'photo': 3, 'art_painting': 4, 'cartoon': 1, 'sketch': 2}
            # selected_domain_dict = {'photo': 2, 'art_painting': 2, 'cartoon': 2, 'sketch': 2}
            # selected_domain_dict = {'photo': 1, 'art_painting': 1, 'cartoon': 1, 'sketch': 1}
            # selected_domain_dict = {'Art': 1, 'Clipart': 3, 'Product': 4, 'RealWorld': 2}  # for office home
            selected_domain_dict = args.selected_domain_dict

            selected_domain_list = []
            for k in selected_domain_dict:
                domain_num = selected_domain_dict[k]
                for i in range(domain_num):
                    selected_domain_list.append(k)

            selected_domain_list = np.random.permutation(selected_domain_list)

            result = Counter(selected_domain_list)
        print(result)

        print(selected_domain_list)

        unlabel_rate = args.unlabel_rate
        total_clients = list(range(args.parti_num))
        # unlabel_clients = random.sample(total_clients, int(unlabel_rate * len(total_clients)))
        unlabel_clients = sample_unlabel_clients(selected_domain_list, total_clients, unlabel_rate)
        label_clients = list(set(total_clients) - set(unlabel_clients))
        print(f'label clients: {label_clients}')
        print(f'unlabel clients: {unlabel_clients}')

        pri_train_loaders, test_loaders = private_dataset.get_data_loaders(label_clients, selected_domain_list)


    else:   # one domain dataset
        unlabel_rate = args.unlabel_rate
        total_clients = list(range(args.parti_num))
        # unlabel_clients = random.sample(total_clients, int(unlabel_rate * len(total_clients)))
        unlabel_clients = sample_unlabel_clients(None, total_clients, unlabel_rate)
        label_clients = list(set(total_clients) - set(unlabel_clients))
        print(f'label clients: {label_clients}')
        print(f'unlabel clients: {unlabel_clients}')

        pri_train_loaders, test_loaders = private_dataset.get_data_loaders(label_clients)


    model.trainloaders = pri_train_loaders
    if hasattr(model, 'ini'):
        model.ini()

    accs_dict = {}
    mean_accs_list = []
    best_acc = 0.0
    Epoch = args.communication_epoch
    pritrain_epoch = args.pritrain_epoch

    accs = global_evaluate(model, test_loaders, private_dataset.SETTING, private_dataset.NAME)
    mean_acc = round(np.mean(accs, axis=0), 3)
    print('Start Accuracy:', str(mean_acc))

    for epoch_index in range(Epoch):
        model.epoch_index = epoch_index
        if hasattr(model, 'loc_update_all'):
            if epoch_index < pritrain_epoch:
                epoch_loc_loss_dict = model.loc_update_label(pri_train_loaders, label_clients)
            else:
                epoch_loc_loss_dict = model.loc_update_all(pri_train_loaders, label_clients, unlabel_clients)
        elif hasattr(model, 'loc_update'):
            epoch_loc_loss_dict = model.loc_update(pri_train_loaders)

        accs = global_evaluate(model, test_loaders, private_dataset.SETTING, private_dataset.NAME)
        mean_acc = round(np.mean(accs, axis=0), 3)
        mean_accs_list.append(mean_acc)
        for i in range(len(accs)):
            if i in accs_dict:
                accs_dict[i].append(accs[i])
            else:
                accs_dict[i] = [accs[i]]
        if mean_acc > best_acc:
            best_acc = mean_acc
            # torch.save(model.global_net.state_dict(), 'warmup/cifar10_fedfix_resnet18.pth')

        print('The ' + str(epoch_index) + ' Communcation Accuracy:', str(mean_acc), ' Best Accuracy:', str(best_acc), 'Method:', model.args.model)
        print(accs)
        print()
        # if epoch_index >= Epoch-1:
        #     tsne_visual(model, test_loaders, private_dataset.SETTING, private_dataset.NAME)
        #     tsne_visual_local(model, test_loaders, args.parti_num, private_dataset.SETTING, private_dataset.NAME)

    if args.csv_log:
        csv_writer.write_acc(accs_dict, mean_accs_list)
