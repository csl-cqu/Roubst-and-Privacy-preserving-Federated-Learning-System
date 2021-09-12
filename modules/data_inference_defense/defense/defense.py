import logging

import torch
import torchvision.transforms as transforms

from modules.data_inference_defense.utils.consts import mnist_mean, mnist_std
from modules.data_inference_defense.utils.policy import *


def get_policy(dataset, model:str, topn=2, augid=-1):
    assert topn > 0
    csvpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_list_{}.csv".format(dataset))
    policy_dict = read_policy_list_from_csv(csvpath)
    use_model = model.lower()

    if use_model not in policy_dict.keys():
        use_model = "default" if "default" in policy_dict.keys() else list(policy_dict.keys())[0]
        logging.warning("Transform policy for {} is not configured, using policy for {}.".format(model, use_model))

    if augid != -1:
        assert augid < len(policy_dict[use_model]), "augid is out of bounds"
        policy_list = policy_dict[use_model][augid:augid+1]
    else:
        policy_list = policy_dict[use_model][:topn]
        assert len(policy_list) > 0, "Not found policy for {}".format(use_model)

        if len(policy_list) < topn:
            logging.warning("Not enough policies, want top %d, only has %d" % (topn, len(policy_list)))

    return PolicyHybrid(policy_list)

class Cutout(object):
    '''Randomly cutout a rect, fill with 0'''
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

#TODO “模型和inversfed输入要求32x32”, 不是32x32的话可以加一个Resize变换

def data_transforms_cifar10_policy(policy: Policy, apply_policy_to_validset=False):
    '''use transform policies in data recovery defense'''
    logging.info("using transform policies in data recovery defense")
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        policy,
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    if apply_policy_to_validset:
        tmp_transform = [policy]
    else:
        tmp_transform = []

    tmp_transform.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    valid_transform = transforms.Compose(tmp_transform)

    return train_transform, valid_transform

def data_transforms_cifar100_policy(policy: Policy, apply_policy_to_validset=False):
    '''use transform policies in data recovery defense'''
    logging.info("using transform policies in data recovery defense")
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        policy,
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    if apply_policy_to_validset:
        tmp_transform = [transforms.ToPILImage(),
                         policy]
    else:
        tmp_transform = []

    tmp_transform.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    valid_transform = transforms.Compose(tmp_transform)

    return train_transform, valid_transform

def data_transforms_mnist_policy(policy: Policy, apply_policy_to_validset=False):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        policy,
        transforms.ToTensor(),
        transforms.Normalize(mnist_mean, mnist_std)
    ])
    if apply_policy_to_validset:
        valid_transform = transforms.Compose([
            transforms.ToPILImage(),
            policy,
            transforms.ToTensor(),
            transforms.Normalize(mnist_mean, mnist_std)
        ])
    else:
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mnist_mean, mnist_std)
        ])
    return train_transform, valid_transform

def data_transforms_mnistRGB_policy(policy: Policy, apply_policy_to_validset=False):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        lambda x: transforms.functional.to_grayscale(x, num_output_channels=3),
        transforms.Resize(32),
        policy,
        transforms.ToTensor(),
        transforms.Normalize(mnist_mean, mnist_std)
    ])
    if apply_policy_to_validset:
        valid_transform = transforms.Compose([
            transforms.ToPILImage(),
            lambda x: transforms.functional.to_grayscale(x, num_output_channels=3),
            transforms.Resize(32),
            policy,
            transforms.ToTensor(),
            transforms.Normalize(mnist_mean, mnist_std)
        ])
    else:
        valid_transform = transforms.Compose([
            transforms.ToPILImage(),
            lambda x: transforms.functional.to_grayscale(x, num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mnist_mean, mnist_std)
        ])
    return train_transform, valid_transform
