import torchvision.transforms as transforms
from utils.conf import data_path
from PIL import Image
from datasets.utils.federated_dataset import FederatedDataset, partition_label_skew_loaders
from datasets.utils.federated_dataset import UnlabeledDataset
import torch.utils.data as data
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
from torchvision import datasets
from backbone.ResNet import resnet10, resnet12, resnet18
from backbone.efficientnet import EfficientNetB0
from backbone.mobilnet_v2 import MobileNetV2
from backbone.autoencoder import autoencoder, mycnn, myvae, mycnn_cls
from torchvision.datasets import MNIST, SVHN, ImageFolder, DatasetFolder, USPS
from torch.utils.data import random_split
from argparse import Namespace
from PIL import Image
import copy
import os

import pdb



class FedLeaCifar10(FederatedDataset):
    NAME = 'fl_cifar10'
    # SETTING = 'domain_skew'
    # DOMAINS_LIST = ['photo', 'art_painting', 'cartoon', 'sketch']
    # percent_dict = {'photo': 0.5, 'art_painting': 0.5, 'cartoon': 0.5, 'sketch': 0.5}

    N_SAMPLES_PER_Class = None
    N_CLASS = 10
    CIFAR10_TRANSFORM = transforms.Compose([
        transforms.Resize((40, 40)),  # Resize the image to 225x225 pixels
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # Normalize using the ImageNet mean and std
                            (0.229, 0.224, 0.225))
    ])

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

    
    def train_test_split(self, domain_dataset):
        train_size = int(0.7 * len(domain_dataset))
        test_size = len(domain_dataset) - train_size
        train_dataset, test_dataset = random_split(domain_dataset, [train_size, test_size])
        train_dataset.data_name = domain_dataset.data_name
        test_dataset.data_name = domain_dataset.data_name
        return train_dataset, test_dataset
    

    def get_data_loaders(self, label_clients):
        # using_list = self.DOMAINS_LIST if len(selected_domain_list) == 0 else selected_domain_list

        nor_transform = self.CIFAR10_TRANSFORM

        train_dataset_list = []
        test_dataset_list = []

        test_transform = transforms.Compose([
            transforms.Resize((40, 40)),  # Resize the image to 225x225 pixels if needed
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize((0.485, 0.456, 0.406),  # Normalize using the ImageNet mean and std
                                (0.229, 0.224, 0.225))
        ])

        # train_dataset_map = {}
        # for _, domain in enumerate(self.DOMAINS_LIST):

        #     domain_dataset = OneDomainDataset(data_name=domain, root=data_path(), transform=nor_transform)
        #     train_dataset, test_dataset = self.train_test_split(domain_dataset)
        #     test_dataset_list.append(test_dataset)
        #     train_dataset_map[domain] = train_dataset
        
        # for i in range(self.args.parti_num):
        #     train_dataset_list.append(datasets.CIFAR10(root=data_path(), transform=nor_transform, train=True, download=True))
        # test_dataset_list.append(datasets.CIFAR10(root=data_path(), transform=nor_transform, train=False, download=True))
        train_dataset = datasets.CIFAR10(root=data_path(), transform=nor_transform, train=True, download=True)
        test_dataset = datasets.CIFAR10(root=data_path(), transform=nor_transform, train=False, download=True)
        # traindls为一个list，长度为客户端数，包含所有客户端的dataloader
        traindls, testdls, net_cls_counts = partition_label_skew_loaders(train_dataset, test_dataset, self)

        return traindls, testdls
    

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), FedLeaCifar10.Nor_TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(args, names_list):
        parti_num = args.parti_num
        nets_dict = {'autoencoder': autoencoder, 'resnet10': resnet10, 'resnet12': resnet12, 'resnet18': resnet18, 'efficient': EfficientNetB0, 'mobilnet': MobileNetV2}
        nets_list = []
        if names_list == None:
            if args.model in ['fedfix']:
                for j in range(parti_num):
                    nets_list.append(resnet18(FedLeaCifar10.N_CLASS))
            else:
                for j in range(parti_num):
                    nets_list.append(mycnn_cls(FedLeaCifar10.N_CLASS))
        else:
            for j in range(parti_num):
                net_name = names_list[j]
                nets_list.append(nets_dict[net_name](FedLeaCifar10.N_CLASS))
        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))
        return transform
