# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# https://github.com/pytorch/vision/blob/8635be94d1216f10fb8302da89233bd86445e449/torchvision/datasets/utils.py

import numpy as np
import torch
import torch 
from torch.utils.model_zoo import tqdm
from torchvision import transforms
from continual_datasets.base_datasets import *
from torch.utils.data import ConcatDataset
from torchvision.datasets import CIFAR10, CIFAR100

def find_tasks_with_unseen_classes(task_id, class_mask):
    seen_classes = set()
    for i in range(task_id + 1):
        seen_classes.update(class_mask[i])

    unseen_tasks = []
    
    for future_task_id in range(task_id + 1, len(class_mask)):
        task_classes = set(class_mask[future_task_id])
        if not task_classes.issubset(seen_classes):
            unseen_tasks.append(future_task_id)
    
    return unseen_tasks

def get_dataset(dataset, transform_train, transform_val, mode, args):
    if dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)
        
    elif dataset == 'FashionMNIST':
        dataset_train = FashionMNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = FashionMNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=True, transform=transform_val)

    elif dataset == 'MNISTM':
        dataset_train = MNISTM(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNISTM(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'SynDigit':
        dataset_train = SynDigit(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = SynDigit(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'KMNIST':
        dataset_train = KMNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = KMNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'PermutedMNIST':
        dataset_train = PermutedMNIST(args.data_path, train=True, download=True, transform=transform_train, random_seed=args.seed)
        dataset_val = PermutedMNIST(args.data_path, train=False, download=True, transform=transform_val, random_seed=args.seed)

    elif dataset == 'NotMNIST':
        dataset_train = NotMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = NotMNIST(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'QMNIST':
        dataset_train = QMNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = QMNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'EMNIST':
        dataset_train = EMNIST_RGB(args.data_path, train=True, download=True, transform=transform_train, num_random_classes=26, split='letters')
        dataset_val = EMNIST_RGB(args.data_path, train=False, download=True, transform=transform_val, num_random_classes=26, split='letters')

    elif dataset == 'Flowers102':
        dataset_train = Flowers102(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = Flowers102(args.data_path, split='val', download=True, transform=transform_val)

    elif dataset == 'StanfordCars':
        dataset_train = StanfordCars(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = StanfordCars(args.data_path, split='test',  download=True, transform=transform_val)

    elif dataset == 'CUB200':
        dataset_train = CUB200(args.data_path, train=True,  download=True, transform=transform_train).data
        dataset_val = CUB200(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'TinyImagenet':
        dataset_train = TinyImagenet(args.data_path, train=True,  download=True, transform=transform_train).data
        dataset_val = TinyImagenet(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'Scene67':
        dataset_train = Scene67(args.data_path, train=True,  download=True, transform=transform_train).data
        dataset_val = Scene67(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'Imagenet_R':
        dataset_train = Imagenet_R(args.data_path, train=True,  download=True, transform=transform_train).data
        dataset_val = Imagenet_R(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'iDigits':
        mnist_train, mnist_val = get_dataset('MNIST', transform_train, transform_val, mode, args)
        svhn_train, svhn_val = get_dataset('SVHN', transform_train, transform_val, mode, args)
        syndigit_train, syndigit_val = get_dataset('SynDigit', transform_train, transform_val, mode, args)
        mnistm_train, mnistm_val = get_dataset('MNISTM', transform_train, transform_val, mode, args)

        dataset_train = ConcatDataset([mnist_train, svhn_train, syndigit_train, mnistm_train])
        dataset_val = ConcatDataset([mnist_val, svhn_val, syndigit_val, mnistm_val])

    elif dataset == 'CORe50':
        dataset_train = CORe50(args.data_path, train=True, download=True, transform=transform_train, mode=mode).data
        dataset_val = CORe50(args.data_path, train=False, download=True, transform=transform_val, mode=mode).data

    elif dataset == 'DomainNet':
        dataset_train = DomainNet(args.data_path, train=True, download=True, transform=transform_train, mode=mode).data
        dataset_val = DomainNet(args.data_path, train=False, download=True, transform=transform_val, mode=mode).data

    elif dataset == 'CLEAR':
        dataset_train = CLEAR(args.data_path, train=True, download=True, transform=transform_train, mode=mode, args=args).data
        dataset_val = CLEAR(args.data_path, train=False, download=True, transform=transform_val, mode=mode, args=args).data

    elif dataset == 'CIFAR10':
        dataset_train = CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = CIFAR10(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'CIFAR100':
        dataset_train = CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = CIFAR100(args.data_path, train=False, download=True, transform=transform_val)
    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    if 1:
        divider = "=" * 60
        print(divider)
        print(f"Dataset: {dataset}")
        # Train dataset 정보 출력
        if isinstance(dataset_train, list):
            total_train = sum(len(ds) for ds in dataset_train)
            print(f"Train dataset total size: {total_train}")
            for i, ds in enumerate(dataset_train):
                try:
                    classes = ds.classes
                    print(f"  Sub-dataset {i}: size {len(ds)}, {len(classes)} classes, classes: {classes}")
                except AttributeError:
                    print(f"  Sub-dataset {i}: size {len(ds)}")
        else:
            print(f"Train dataset size: {len(dataset_train)}")
            try:
                print(f"Number of classes: {len(dataset_train.classes)}")
                print(f"Classes: {dataset_train.classes}")
            except AttributeError:
                pass

        # Validation dataset 정보 출력
        if isinstance(dataset_val, list):
            total_val = sum(len(ds) for ds in dataset_val)
            print(f"Validation dataset total size: {total_val}")
            for i, ds in enumerate(dataset_val):
                try:
                    classes = ds.classes
                    print(f"  Sub-dataset {i}: size {len(ds)}, {len(classes)} classes, classes: {classes}")
                except AttributeError:
                    print(f"  Sub-dataset {i}: size {len(ds)}")
        else:
            print(f"Validation dataset size: {len(dataset_val)}")
            try:
                print(f"Number of classes: {len(dataset_val.classes)}")
                print(f"Classes: {dataset_val.classes}")
            except AttributeError:
                pass
        print(divider)
    
    return dataset_train, dataset_val

def get_ood_dataset(dataset_name, args):
    if 1:
        print(f"Loading OOD dataset: {dataset_name}")
    dataset = get_dataset(dataset_name, transform_train=build_transform(True,args), transform_val=build_transform(False,args), mode='joint', args=args)[0]
    ood_dataset = UnknownWrapper(dataset, args.num_classes)
    return ood_dataset

def set_data_config(args):
    if args.dataset == "iDigits":
        args.num_classes = 10
        args.num_domains = 4
        args.id_datasets = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
    elif args.dataset == "DomainNet":
        args.num_classes = 345
        args.num_domains = 6
    elif args.dataset == "CORe50":
        args.num_classes = 50
        args.num_domains = 8 
    elif args.dataset == "CLEAR":
        args.num_classes = 100
        args.num_domains = 5 #11    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return args

def build_transform(is_train, args):
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    return transform

class UnknownWrapper(torch.utils.data.Dataset):
    """
    원본 데이터셋의 라벨을 모두 unknown_label(= num_known)로 변경합니다.
    """
    def __init__(self, dataset, unknown_label):
        self.dataset = dataset
        self.unknown_label = unknown_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, _ = self.dataset[index]
        
        return x, self.unknown_label

class RandomSampleWrapper(torch.utils.data.Dataset):
    """
    주어진 데이터셋에서 num_samples만큼 랜덤으로 샘플링하여 반환합니다.
    """
    def __init__(self, dataset, num_samples, seed):
        self.dataset = dataset
        self.num_samples = num_samples
        np.random.seed(seed)
        # replacement 없이 num_samples 개의 인덱스 선택
        self.indices = np.random.choice(len(dataset), size=num_samples, replace=False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


