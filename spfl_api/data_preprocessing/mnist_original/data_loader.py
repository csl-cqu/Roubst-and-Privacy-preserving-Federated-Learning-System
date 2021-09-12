import gzip
import os

import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
from torchvision import datasets

from modules.data_inference_defense.defense import data_transforms_mnist_policy


class MNIST_ORIGINAL(data.Dataset):
    def __init__(self, folder, data_name, label_name, transform=None, dataidxs=None):
        self.data, self.target = load_data(folder, data_name, label_name)  # type: np.ndarray

        if dataidxs is not None:
            self.data, self.target = self.data[dataidxs], self.target[dataidxs]

        self.data, self.target = np.asarray(self.data).reshape((-1, 28, 28, 1)), np.asarray(self.target)

        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], int(self.target[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


def load_data(data_folder, data_name, label_name):
    with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:  # rb表示的是读取二进制数据
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return (x_train, y_train)


def _data_transforms_mnist():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066373765468597,), (0.30810782313346863,))
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066373765468597,), (0.30810782313346863,))
    ])
    return train_transform, valid_transform


def get_dataloader_MNIST_ORIGINAL(datadir, train_bs, test_bs, transform_policy=None, apply_policy_to_validset=False,
                                  download=False):
    if not transform_policy:
        transform_train, transform_test = _data_transforms_mnist()
    else:
        transform_train, transform_test = data_transforms_mnist_policy(transform_policy, apply_policy_to_validset)

    if download:
        train_dataset = datasets.MNIST(root=datadir, train=True, transform=transform_train, download=download)
        test_dataset = datasets.MNIST(root=datadir, train=False, transform=transform_test, download=download)
    else:
        train_dataset = MNIST_ORIGINAL(datadir, "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                                       transform=transform_train)
        test_dataset = MNIST_ORIGINAL(datadir, "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
                                      transform=transform_test)

    # 载入数据集
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True, drop_last=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=test_bs, shuffle=True, drop_last=True)

    return train_loader, test_loader
