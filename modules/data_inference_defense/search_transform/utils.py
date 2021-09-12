import os
import sys

from torch.utils import data
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from spfl_api.utils.pathfinder import *


def pretrained_model_path(dataset, model, epochs, assert_path=False):
    base_dir = os.path.join(MODULE_DATA_RECOV,"search_transform", "pretrained")
    save_dir = 'dataset_{}_model_{}'.format(dataset, model)
    save_dir = os.path.join(base_dir, save_dir)
    if assert_path:
        assert os.path.exists(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, "{}.pth".format(epochs))
    return model_path

def pri_score_path(dataset, model, epochs, assert_path=False):
    base_dir = os.path.join(MODULE_DATA_RECOV,"search_transform", "Spri")
    pri_path = 'dataset_{}_model_{}_epochs_{}'.format(dataset, model, epochs)
    pri_path = os.path.join(base_dir, pri_path)
    if assert_path:
        assert os.path.exists(pri_path)

    if not os.path.exists(pri_path):
        os.makedirs(pri_path)

    return pri_path

def acc_score_path(dataset, model, epochs, assert_path=False):
    base_dir = os.path.join(MODULE_DATA_RECOV,"search_transform", "Sacc")
    acc_path = 'dataset_{}_model_{}_epochs_{}'.format(dataset, model, epochs)
    acc_path = os.path.join(base_dir, acc_path)
    if assert_path:
        assert os.path.exists(acc_path)

    if not os.path.exists(acc_path):
        os.makedirs(acc_path)

    return acc_path

def result_path(dataset, model, epochs):
    base_dir = os.path.join(MODULE_DATA_RECOV,"search_transform", "search_result")
    res_path = 'dataset_{}_model_{}_epochs_{}.csv'.format(dataset, model, epochs)
    res_path = os.path.join(base_dir, res_path)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    return res_path

def aug_list_gen_path(dataset, model, epochs):
    base_dir = os.path.join(MODULE_DATA_RECOV,"search_transform", "calc_policy_score")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # if sys.platform == 'win32':
    #     aug_list_path = 'dataset_{}_model_{}_epochs{}.bat'.format(dataset, model, epochs)
    # else:
    aug_list_path = 'dataset_{}_model_{}_epochs{}.sh'.format(dataset, model, epochs)
    aug_list_path = os.path.join(base_dir, aug_list_path)

    return aug_list_path

class MNIST(data.Dataset):
    '''This dataset is for '''
    def __init__(self, x, y, transform=None):
        (self.data, self.target) = x, y
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], int(self.target[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


