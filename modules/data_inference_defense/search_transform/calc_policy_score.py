import os
import sys
import torch

sys.path.insert(0, './')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
seed=23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import random
random.seed(seed)

import numpy as np
import argparse
import torch.nn.functional as F

from modules.data_inference_defense.utils.loss import Classification
from modules.data_inference_defense.utils import consts
from modules.data_inference_defense.utils.policy import make_policy_hybrid, make_policy_single
from modules.data_inference_defense.search_transform.utils import acc_score_path, pri_score_path, pretrained_model_path
from spfl_api.utils.pathfinder import *
from spfl_api.data_preprocessing.mnistRGB.data_loader import get_dataloader_mnistRGB
from spfl_api.data_preprocessing.cifar10.data_loader import get_dataloader_CIFAR10
from spfl_api.data_preprocessing.cifar100.data_loader import get_dataloader_CIFAR100
from spfl_experiments.distributed.fedavg.main_fedavg import create_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


parser = argparse.ArgumentParser(description='Calculate scores on a given aug_list.')
parser.add_argument('--model', default="convnet", type=str, help='Pretrained model.')
parser.add_argument('--dataset', default="cifar100", type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=100, type=int, help='Pretrained model\'s epoch.')
parser.add_argument('--aug_list', required=True, type=str, help='indices like 1-2-3')
args = parser.parse_args()

# trained_model = True
# num_images = 1


def eval_score(jacob:np.ndarray, labels=None):
    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1./(v + k))


def get_batch_jacobian(net, x, target):
    net.eval()
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach()

def calculate_dw(model, inputs, labels, loss_fn):
    model.zero_grad()
    target_loss, _, _ = loss_fn(model(inputs), labels)
    dw = torch.autograd.grad(target_loss, model.parameters())
    return dw

# GradSim
def cal_dis(a, b, metric='L2'):
    a, b = a.flatten(), b.flatten()
    if metric == 'L2':
        return torch.mean((a - b) * (a - b)).item()
    elif metric == 'L1':
        return torch.mean(torch.abs(a-b)).item()
    elif metric == 'cos':
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    else:
        raise NotImplementedError



def accuracy_metric(idx_list, model, trainloader, device):

    # prepare data
    ground_truth, labels = [], []
    for idx in idx_list:
        img, label = trainloader.dataset[idx]
        idx += 1
        if label not in labels:
            labels.append(torch.as_tensor((label,), device=device))
            ground_truth.append(img.to(device))

    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels)
    model.zero_grad()
    jacobs, labels= get_batch_jacobian(model, ground_truth, labels)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
    return eval_score(jacobs, labels)


# get Spri
def privacy_metric(idx, model, loss_fn, trainloader, device):
    '''only one image at a time (D==1)'''
    if args.dataset == 'cifar10':
        dm = torch.as_tensor(consts.cifar10_mean, dtype=torch.float, device=device)[:, None, None]
        ds = torch.as_tensor(consts.cifar10_std, dtype=torch.float, device=device)[:, None, None]
    elif args.dataset == 'cifar100':
        dm = torch.as_tensor(consts.cifar100_mean, dtype=torch.float, device=device)[:, None, None]
        ds = torch.as_tensor(consts.cifar100_std, dtype=torch.float, device=device)[:, None, None]
    elif 'mnist' in args.dataset:
        dm = torch.as_tensor(consts.mnist_mean, dtype=torch.float, device=device)[:, None, None]
        ds = torch.as_tensor(consts.mnist_std, dtype=torch.float, device=device)[:, None, None]
    else:
        raise NotImplementedError
    
    # prepare data
    ground_truth, labels = [], []
    img, label = trainloader.dataset[idx]
    labels.append(torch.as_tensor((label,), device=device))
    ground_truth.append(img.to(device))

    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels).long()
    model.zero_grad()
    # calcuate ori dW
    target_loss, _, _ = loss_fn(model(ground_truth), labels)
    input_gradient = torch.autograd.grad(target_loss, model.parameters())

    metric = 'cos'

    model.eval()
    dw_list = list()
    dx_list = list()
    # ground_truth == x
    # num_images == D = 1
    bin_num = 20 # == K
    # noise_input == x0
    noise_input = (torch.rand((ground_truth.shape)).cuda() - dm) / ds
    for dis_iter in range(bin_num+1):
        # dis_iter/bin_num == i
        model.zero_grad()
        # fake_ground_truth = x'(i) = i*x + 1-i*x0
        fake_ground_truth = (1.0 / bin_num * dis_iter * ground_truth + 1. / bin_num * (bin_num - dis_iter) * noise_input).detach()
        fake_dw = calculate_dw(model, fake_ground_truth, labels, loss_fn)
        # gradsim(fake_dw, true_w)
        dw_loss = sum([cal_dis(dw_a, dw_b, metric=metric) for dw_a, dw_b in zip(fake_dw, input_gradient)]) / len(input_gradient)

        dw_list.append(dw_loss)

    # PSNR(x0, x)?
    interval_distance = cal_dis(noise_input, ground_truth, metric='L1') / bin_num

    def area_ratio(y_list, inter):
        area = 0
        max_area = inter * bin_num
        for idx in range(1, len(y_list)):
            prev = y_list[idx-1]
            cur = y_list[idx]
            area += (prev + cur) * inter / 2
        return area / max_area

    return area_ratio(dw_list, interval_distance)



def main(args):
    assert args.aug_list
    skip_acc = False
    skip_pri = False
    # Sacc
    root_dir = acc_score_path(args.dataset, args.model, args.epochs, assert_path=False)
    pathname_acc = os.path.join(root_dir, f'{args.aug_list}.npy')
    if os.path.exists(pathname_acc):
        print('Sacc exists')
        skip_acc = True

    # Spri
    root_dir = pri_score_path(args.dataset, args.model, args.epochs, assert_path=False)
    pathname_pri = os.path.join(root_dir, f'{args.aug_list}.npy')
    if os.path.exists(pathname_pri):
        print('Spri exists')
        skip_pri = True

    if skip_acc and skip_pri:
        exit(0)

    if '+' in args.aug_list:
        policy = make_policy_hybrid(args.aug_list.split('+'))
    else:
        policy = make_policy_single(args.aug_list)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    datadir = os.path.join(DATA, args.dataset)
    if args.dataset == 'mnistRGB':
        datadir = os.path.join(DATA, 'mnist_original')
    batch_size = 128
    if not os.path.exists(datadir):
        raise NotImplementedError

    if args.dataset == 'mnistRGB':
        trainloader, _, _, _ = get_dataloader_mnistRGB(batch_size, datadir, transform_policy=policy,
                                                 apply_policy_to_validset=False)
        n_class = 10
        loss_fn = Classification()
    elif args.dataset == 'cifar10':
        trainloader, _ = get_dataloader_CIFAR10(datadir, batch_size, batch_size, transform_policy=policy,
                                                apply_policy_to_validset=False)
        n_class = 10
        loss_fn = Classification()
    elif args.dataset == 'cifar100':
        trainloader, _ = get_dataloader_CIFAR100(datadir, batch_size, batch_size, transform_policy=policy,
                                                 apply_policy_to_validset=False)
        n_class = 100
        loss_fn = Classification()
    else:
        raise NotImplementedError

    model = create_model(args, args.model, n_class)
    model.to(device)
    model.eval()

    print("calculate " + args.aug_list)

    import time
    start = time.time()

    # Sacc
    acc_score_list = list()
    for run in range(10):
        large_sample_list = [200 + run * 100 + i for i in range(100) for run in range(10)]
        # Sacc
        acc_score = accuracy_metric(large_sample_list, model, trainloader, device)
        acc_score_list.append(acc_score)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    np.save(pathname_acc, acc_score_list)
    print('saving. avg acc_score: {}'.format(np.mean(acc_score_list)))

    # Spri
    model_path = pretrained_model_path(args.dataset, args.model, args.epochs, assert_path=True)
    model.load_state_dict(torch.load(model_path))
    sample_list = [200+i*5 for i in range(100)]
    pri_score_list = list()
    for valid_id, idx in enumerate(sample_list):
        # Spri
        # use pic idx
        pri_score = privacy_metric(idx, model, loss_fn, trainloader, device)
        pri_score_list.append(pri_score)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if len(pri_score_list) > 0:
        print('saving. avg pri_score: {}'.format(np.mean(pri_score_list)))
        np.save(pathname_pri, pri_score_list)
    else:
        print('len(pri_score_list) == 0, not saving')

    print('time cost ', time.time() - start)
    print('\n')

if __name__ == '__main__':
    main(args)
