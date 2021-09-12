import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


import torch
import torchvision
from modules.data_inference_defense.utils import consts
from modules.data_inference_defense.attack import attack_lib
from modules.data_inference_defense.search_transform.utils import *
from spfl_experiments.distributed.fedavg.main_fedavg import create_model
from modules.data_inference_defense.utils.loss import Classification
from modules.data_inference_defense.utils.policy import make_policy_hybrid, make_policy_single
from spfl_api.data_preprocessing.cifar10.data_loader import get_dataloader_CIFAR10
from spfl_api.data_preprocessing.cifar100.data_loader import get_dataloader_CIFAR100
from spfl_api.data_preprocessing.mnistRGB.data_loader import get_dataloader_mnistRGB
from spfl_api.utils.pathfinder import *

sys.path.insert(0, './')
seed=23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import random
random.seed(seed)

import numpy as np
import argparse

def create_config(optim):
    print(optim)
    if optim == 'inversed':
        config = dict(signed=True,
                boxed=True,
                cost_fn='sim',
                indices='def',
                weights='equal',
                lr=0.1,
                optim='adam',
                restarts=1,
                max_iterations=4800,
                total_variation=1e-4,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss')
    elif optim == 'inversed-zero':
        config = dict(signed=True,
            boxed=True,
            cost_fn='sim',
            indices='def',
            weights='equal',
            lr=0.1,
            optim='adam',
            restarts=1,
            max_iterations=4800,
            total_variation=1e-4,
            init='zeros',
            filter='none',
            lr_decay=True,
            scoring_choice='loss')
    elif optim == 'inversed-sim-out':
        config = dict(signed=True,
            boxed=True,
            cost_fn='out_sim',
            indices='def',
            weights='equal',
            lr=0.1,
            optim='adam',
            restarts=1,
            max_iterations=4800,
            total_variation=1e-4,
            init='zeros',
            filter='none',
            lr_decay=True,
            scoring_choice='loss')
    elif optim == 'inversed-sgd-sim':
        config = dict(signed=True,
                boxed=True,
                cost_fn='sim',
                indices='def',
                weights='equal',
                lr=0.1,
                optim='sgd',
                restarts=1,
                max_iterations=4800,
                total_variation=1e-4,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss')
    elif optim == 'inversed-LBFGS-sim':
        config = dict(signed=True,
                boxed=True,
                cost_fn='sim',
                indices='def',
                weights='equal',
                lr=1e-4,
                optim='LBFGS',
                restarts=16,
                max_iterations=300,
                total_variation=1e-4,
                init='randn',
                filter='none',
                lr_decay=False,
                scoring_choice='loss')
    elif optim == 'inversed-adam-L1':
        config = dict(signed=True,
                boxed=True,
                cost_fn='l1',
                indices='def',
                weights='equal',
                lr=0.1,
                optim='adam',
                restarts=1,
                max_iterations=4800,
                total_variation=1e-4,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss')
    elif optim == 'inversed-adam-L2':
        config = dict(signed=True,
                boxed=True,
                cost_fn='l2',
                indices='def',
                weights='equal',
                lr=0.1,
                optim='adam',
                restarts=1,
                max_iterations=4800,
                total_variation=1e-4,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss')
    elif optim == 'zhu':
        config = dict(signed=False,
                        boxed=False,
                        cost_fn='l2',
                        indices='def',
                        weights='equal',
                        lr=1e-4,
                        optim='LBFGS',
                        restarts=2,
                        max_iterations=50, # ??
                        total_variation=1e-3,
                        init='randn',
                        filter='none',
                        lr_decay=False,
                        scoring_choice='loss')
        # seed=1
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # import random
        # random.seed(seed)
    else:
        raise NotImplementedError
    return config

parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--model', required=True, type=str, help='Vision model.')
parser.add_argument('--dataset', required=True, type=str, help='Vision dataset.')
parser.add_argument('--optim', required=True, type=str, help='Optimization method.')
parser.add_argument("--mode", required=True, type=str, choices=['normal', 'policy'], help='use transform policy or not')
parser.add_argument('--aug_list', default='', type=str, help='Augmentation indices connected by -, e.g. 1-2-3')
parser.add_argument('--rlabel', default=False, action='store_true', help='reconstruct label')
parser.add_argument('-p', '--use_pretrained_model', default=False, action='store_true', help='use pretrained model in ./search_transform/pretrained. Turn this on (maybe) result in smaller psnr value (worse attack result)')
parser.add_argument('--epochs', default=0, type=int, help="Pretrained model's epochs")
parser.add_argument('--hide_output_img', default=False, action='store_true')

args = parser.parse_args()

num_images = 1 # number of reconstruct images

def reconstruct(idx, model, loss_fn, validloader, config, args, device):
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
    # while len(labels) < num_images:
    img, label = validloader.dataset[idx]
        # idx += 1
    # if label not in labels:
    labels.append(torch.as_tensor((label,), device=device))
    ground_truth.append(img.to(device))

    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels).long()
    model.zero_grad()
    target_loss, _, _ = loss_fn(model(ground_truth), labels)
    param_list = [param for param in model.parameters() if param.requires_grad]
    input_gradient = torch.autograd.grad(target_loss, param_list)


    # validation
    print('ground truth label is ', labels)
    rec_machine = attack_lib.GradientReconstructor(model, (dm, ds), config, num_images=num_images)
    if args.dataset == 'cifar10':
        shape = (3, 32, 32)
    elif args.dataset == 'cifar100':
        shape = (3, 32, 32)
    elif args.dataset == 'mnistRGB':
        shape = (3, 32, 32)
    else:
        raise NotImplementedError

    if args.rlabel:
        # reconstruction label
        output, stats = rec_machine.reconstruct(input_gradient, None, img_shape=shape)
    else:
        # specify label
        output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=shape)

    output_denormalized = output * ds + dm
    input_denormalized = ground_truth * ds + dm
    mean_loss = torch.mean((input_denormalized - output_denormalized) * (input_denormalized - output_denormalized))
    print("after optimization, the true mse loss {}".format(mean_loss))

    if not args.hide_output_img:
        save_dir = attack_output_dir(args.dataset, args.model, args.optim, args.mode, args.aug_list, args.rlabel, args.use_pretrained_model, args.epochs)
        rec_save_dir = os.path.join(save_dir, "image", "rec")
        ori_save_dir = os.path.join(save_dir, "image", "ori")
        if not os.path.exists(rec_save_dir):
            os.makedirs(rec_save_dir)
        if not os.path.exists(ori_save_dir):
            os.makedirs(ori_save_dir)

        torchvision.utils.save_image(output_denormalized.cpu().clone(), '{}/{}.jpg'.format(rec_save_dir, idx))
        torchvision.utils.save_image(input_denormalized.cpu().clone(), '{}/{}.jpg'.format(ori_save_dir, idx))

    test_mse = (output_denormalized.detach() - input_denormalized).pow(2).mean().cpu().detach().numpy()
    feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()
    test_psnr = attack_lib.metrics.psnr(output_denormalized, input_denormalized)

    return {
        'idx': idx,
        'test_mse': test_mse,
        'feat_mse': feat_mse,
        'test_psnr': test_psnr
    }, test_psnr

def attack_output_dir(dataset, model, optim, mode, aug_list, rlabel, use_pretrained_model, epochs):
    base_dir = os.path.join(MODULE_DATA_RECOV, "attack", "output")
    output_path =  'dataset_{}_model_{}_optim_{}_mode_{}_rlabel_{}'.format(dataset, model, optim, mode, rlabel)

    if use_pretrained_model:
        output_path += '_pretrainedEpoch_{}'.format(epochs)

    if mode == 'policy':
        output_path += '_auglist_{}'.format(aug_list)

    output_path = os.path.join(base_dir, output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    return output_path

def main():
    config = create_config(args.optim)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # config['max_iterations'] = 10
    batch_size = 128

    datadir = os.path.join(DATA, args.dataset)
    if args.dataset == 'mnistRGB':
        datadir = os.path.join(DATA, 'mnist_original')
    batch_size = 128
    if not os.path.exists(datadir):
        raise NotImplementedError

    if args.aug_list:
        if '+' in args.aug_list:
            policy = make_policy_hybrid(args.aug_list.split('+'))
        else:
            policy = make_policy_single(args.aug_list)
    else:
        policy = None

    if args.dataset == 'mnistRGB':
        trainloader, validloader, _, _ = get_dataloader_mnistRGB(batch_size, datadir, transform_policy=policy,
                                                       apply_policy_to_validset=args.mode=='policy')
        n_class = 10
        loss_fn = Classification()
    elif args.dataset == 'cifar10':
        trainloader, validloader = get_dataloader_CIFAR10(datadir, batch_size, batch_size, transform_policy=policy,
                                                apply_policy_to_validset=args.mode=='policy')
        n_class = 10
        loss_fn = Classification()
    elif args.dataset == 'cifar100':
        trainloader, validloader = get_dataloader_CIFAR100(datadir, batch_size, batch_size, transform_policy=policy,
                                                 apply_policy_to_validset=args.mode=='policy')
        n_class = 100
        loss_fn = Classification()
    else:
        raise NotImplementedError

    model = create_model(args, args.model, n_class)
    model.to(device)

    if args.use_pretrained_model:
        model_path = pretrained_model_path(args.dataset, args.model, args.epochs, assert_path=True)
        model.load_state_dict(torch.load(model_path))
        print('load model successful')

    if args.rlabel:
        for name, param in model.named_parameters():
            if 'fc' in name:
                param.requires_grad = False

    model.eval()
    sample_list = [i for i in range(10)]
    psnr_list = list()
    for valid_id, idx in enumerate(sample_list):
        if args.mode == 'normal':
            print('reconstruct {}th img'.format(idx))
        else:
            print('reconstruct {}th img with policy {}'.format(idx, args.aug_list))
        _, psnr = reconstruct(idx, model, loss_fn, validloader, config, args, device)
        psnr_list.append(psnr)
        torch.cuda.empty_cache()

    save_dir = attack_output_dir(args.dataset, args.model, args.optim, args.mode, args.aug_list, args.rlabel, args.use_pretrained_model, args.epochs)
    np.save('{}/psnr_{}.npy'.format(save_dir, args.aug_list), psnr_list)
    avg_psnr = np.mean(psnr_list)

    with open('{}/psnr_{}.csv'.format(save_dir, args.aug_list), 'w') as f:
        header = 'avg_psnr'
        f.write(header)
        f.write('\n')
        f.write(str(avg_psnr))
        f.close()

    print("reconstruct finish")

if __name__ == '__main__':
    main()
