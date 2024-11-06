import torchvision.transforms as transforms
from utils.conf import data_path
from PIL import Image
from datasets.utils.federated_dataset import FederatedDataset, partition_pacs_domain_skew_loaders
from datasets.utils.federated_dataset import UnlabeledDataset
import torch.utils.data as data
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
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


class MyDigits(data.Dataset):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=True, data_name=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.data_name == 'mnist':
            dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)
        elif self.data_name == 'usps':
            dataobj = USPS(self.root, self.train, self.transform, self.target_transform, self.download)
        elif self.data_name == 'svhn':
            if self.train:
                dataobj = SVHN(self.root, 'train', self.transform, self.target_transform, self.download)
            else:
                dataobj = SVHN(self.root, 'test', self.transform, self.target_transform, self.download)
        return dataobj

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        img, target = self.dataset[index]
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class ImageFolder_Custom(DatasetFolder):
    def __init__(self, data_name, root, train=True, transform=None, target_transform=None):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.imagefolder_obj = ImageFolder(self.root + self.data_name + '/train/', self.transform, self.target_transform)
        else:
            self.imagefolder_obj = ImageFolder(self.root + self.data_name + '/val/', self.transform, self.target_transform)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


# Dataset with one domain data
class OneDomainDataset(data.Dataset):
    def __init__(self, data_name, root, train=True, transform=None, target_transform=None):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.domain_map = {'cartoon': 0, 'art_painting': 1, 'photo': 2, 'sketch': 3}

        fulldata = []
        label_name = []
        fullconcept = []
        txt_path = os.path.join(self.root, self.data_name + '.txt')

        images, labels = self._dataset_info(txt_path)
        concept = [self.domain_map[self.data_name]] * len(labels)
        fulldata.extend(images)
        label_name.extend(labels)
        fullconcept.extend(concept)

        classes = list(set(label_name))
        classes.sort()

        class_to_idx = {classes[i]: i for i in range(len(classes))}

        fulllabel = [class_to_idx[x] for x in label_name]

        self.data = fulldata
        self.label = fulllabel
        self.concept = fullconcept
    

    def __getitem__(self, index):
        data, label, concept = self.data[index], self.label[index], self.concept[index]
        _img = Image.open(data).convert('RGB')
        img = self.transform(_img)
        return img, label
    

    def __len__(self):
        return len(self.data)
    
    
    def _dataset_info(self, txt_labels):
        with open(txt_labels, 'r') as f:
            images_list = f.readlines()

        file_names = []
        labels = []
        for row in images_list:
            row = row.split(' ')
            path = os.path.join(self.root, row[0])
            path = path.replace('\\', '/')

            file_names.append(path)
            labels.append(int(row[1]))

        return file_names, labels



class FedLeaPACS(FederatedDataset):
    NAME = 'fl_pacs'
    # SETTING = 'domain_skew'
    DOMAINS_LIST = ['photo', 'art_painting', 'cartoon', 'sketch']
    percent_dict = {'photo': 0.5, 'art_painting': 0.5, 'cartoon': 0.5, 'sketch': 0.5}

    N_SAMPLES_PER_Class = None
    N_CLASS = 7
    PACS_TRANSFORM = transforms.Compose([
        transforms.Resize((225, 225)),  # Resize the image to 225x225 pixels
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # Normalize using the ImageNet mean and std
                            (0.229, 0.224, 0.225))
    ])


    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        assert(FedLeaPACS.percent_dict.keys() == args.selected_domain_dict.keys())
        for key in FedLeaPACS.percent_dict:
            FedLeaPACS.percent_dict[key] = 1 / args.selected_domain_dict[key]

    
    def train_test_split(self, domain_dataset):
        train_size = int(0.7 * len(domain_dataset))
        test_size = len(domain_dataset) - train_size
        train_dataset, test_dataset = random_split(domain_dataset, [train_size, test_size])
        train_dataset.data_name = domain_dataset.data_name
        test_dataset.data_name = domain_dataset.data_name
        return train_dataset, test_dataset
    

    def get_data_loaders(self, label_clients, selected_domain_list=[]):
        using_list = self.DOMAINS_LIST if len(selected_domain_list) == 0 else selected_domain_list

        nor_transform = self.PACS_TRANSFORM

        train_dataset_list = []
        test_dataset_list = []

        test_transform = transforms.Compose([
            transforms.Resize((225, 225)),  # Resize the image to 225x225 pixels if needed
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize((0.485, 0.456, 0.406),  # Normalize using the ImageNet mean and std
                                (0.229, 0.224, 0.225))
        ])

        train_dataset_map = {}
        for _, domain in enumerate(self.DOMAINS_LIST):

            domain_dataset = OneDomainDataset(data_name=domain, root=data_path(), transform=nor_transform)
            train_dataset, test_dataset = self.train_test_split(domain_dataset)
            test_dataset_list.append(test_dataset)
            train_dataset_map[domain] = train_dataset
        
        for i, domain in enumerate(using_list):
            train_dataset_list.append(copy.deepcopy(train_dataset_map[domain]))
        # traindls为一个list，长度为客户端数，包含所有客户端的dataloader
        traindls, testdls = partition_pacs_domain_skew_loaders(train_dataset_list, test_dataset_list, self)

        return traindls, testdls
    

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), FedLeaPACS.Nor_TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(args, names_list):
        parti_num = args.parti_num
        nets_dict = {'autoencoder': autoencoder, 'resnet10': resnet10, 'resnet12': resnet12, 'resnet18': resnet18, 'efficient': EfficientNetB0, 'mobilnet': MobileNetV2}
        nets_list = []
        if names_list == None:
            if args.model in ['fedfix']:
                for j in range(parti_num):
                    nets_list.append(mycnn(FedLeaPACS.N_CLASS))
            else:
                for j in range(parti_num):
                    nets_list.append(mycnn_cls(FedLeaPACS.N_CLASS))
        else:
            for j in range(parti_num):
                net_name = names_list[j]
                nets_list.append(nets_dict[net_name](FedLeaPACS.N_CLASS))
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
