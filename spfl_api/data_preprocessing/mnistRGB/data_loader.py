import os

import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms

from modules.data_inference_defense.defense import data_transforms_mnistRGB_policy
from spfl_api.data_preprocessing.mnist_original.data_loader import MNIST_ORIGINAL

# datasrc: mnist_original

def _data_transforms_mnistRGB():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        lambda x: transforms.functional.to_grayscale(x, num_output_channels=3),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.13066373765468597,), (0.30810782313346863,))
    ])
    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        lambda x: transforms.functional.to_grayscale(x, num_output_channels=3),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.13066373765468597,), (0.30810782313346863,))
    ])
    return train_transform, valid_transform

def load_partition_data_mnistRGB(batch_size, partition_method, partition_alpha, client_number,
                              datadir="./../../../data/mnist_original",
                              transform_policy=None, apply_policy_to_validset=False):

    train_data_global, test_data_global, train_dataset, test_dataset = get_dataloader_mnistRGB(batch_size, datadir, transform_policy, apply_policy_to_validset)

    net_dataidx_map = partition_data(train_dataset.data.shape[0], partition_method, client_number, partition_alpha)

    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    test_data_num = len(test_data_global)

    # get local dataset
    train_data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        train_data_local_num_dict[client_idx] = local_data_num

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local, _, _ = get_dataloader_mnistRGB(batch_size, datadir, transform_policy, apply_policy_to_validset, dataidxs=dataidxs)
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    class_num = 10

    return train_data_num, test_data_num, train_data_global, test_data_global, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


def get_dataloader_mnistRGB(batch_size,
                            datadir="./../../../data/mnist_original",
                            transform_policy=None, apply_policy_to_validset=False, dataidxs=None):

    if transform_policy is None:
        transform_train, transform_test = _data_transforms_mnistRGB()
    else:
        transform_train, transform_test = data_transforms_mnistRGB_policy(transform_policy, apply_policy_to_validset)

    train_dataset = MNIST_ORIGINAL(datadir, "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                                   transform=transform_train, dataidxs=dataidxs)
    test_dataset = MNIST_ORIGINAL(datadir, "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
                                  transform=transform_test)
    # transform to batches
    # train_batch = batch_data(train_dataset.data, batch_size)
    # test_batch = batch_data(test_dataset.data, batch_size)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader, train_dataset, test_dataset

def partition_data(n_train, partition, n_nets, alpha):

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = n_train
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(n_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map

