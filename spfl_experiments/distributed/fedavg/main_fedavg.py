import argparse
import logging
import os
import random
import socket
import sys
import traceback

import numpy as np
import setproctitle
import torch
import wandb
from mpi4py import MPI

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from modules.data_inference_defense.defense import get_policy
from spfl_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file
from spfl_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist
from spfl_api.data_preprocessing.mnistRGB.data_loader import load_partition_data_mnistRGB

from spfl_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from spfl_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10

from spfl_api.model.cv.cnn import ConvNet, CNN_OriginalFedAvg
from spfl_api.model.cv.resnet_gn import resnet18
from spfl_api.model.cv.mobilenet import mobilenet
from spfl_api.model.cv.resnet import resnet56
from spfl_api.model.linear.lr import LogisticRegression
from spfl_api.model.cv.mobilenet_v3 import MobileNetV3
from spfl_api.model.cv.convnet_test import SampleConvNet

from spfl_api.distributed.fedavg.FedAvgAPI import FedML_init, FedML_FedAvg_distributed


def add_args(parser):
    parser.add_argument('--model', type=str, default='mobilenet', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--client_num_in_total', type=int, default=1000, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=4, metavar='NN',
                        help='number of workers')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--backend', type=str, default="MPI",
                        help='Backend for Server and Client')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu_server_num', type=int, default=1,
                        help='gpu_server_num')

    parser.add_argument('--gpu_num_per_server', type=int, default=4,
                        help='gpu_num_per_server')

    parser.add_argument('--gpu_mapping_file', type=str, default="gpu_mapping.yaml",
                        help='the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.')

    parser.add_argument('--gpu_mapping_key', type=str, default="mapping_default",
                        help='the key in gpu utilization file')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument("--wandb-off", action="store_true")
    parser.add_argument('--wandb-name', type=str, default='', help='name of this run for www.wandb.com')
    parser.add_argument('--no_save_to_localfile', action="store_true",
                        help="should save model param to local file or not")
    parser.add_argument('--save_path', type=str, default='', help='full pathname for model localfile saving')

    data_recvry_protection = parser.add_argument_group("module for data recovery protection")

    data_recvry_protection.add_argument('--data_recovery_protection', action="store_true",
                                        help='turn preprocess data on to protect data being recovered')

    data_recvry_protection.add_argument("--topn", type=int, default=2, metavar='N',
                                        help='top n hybrid policies used in policy list')

    data_recvry_protection.add_argument("--augid", type=int, default=-1, metavar='N',
                                        help='policy id to be used in policy list')

    diff_privacy = parser.add_argument_group("module for differential privacy")

    diff_privacy.add_argument('--differential_privacy', action="store_true", help='')
    diff_privacy.add_argument('--dp_sigma', type=float, help='')
    diff_privacy.add_argument('--dp_delta', type=float, help='')
    diff_privacy.add_argument('--grad_norm', type=float, help='')
    diff_privacy.add_argument('--separate_client', action="store_true")

    parser.add_argument('--use_gradient', action="store_true", help='whether to enable client gradients')

    byzantine_aggregate = parser.add_argument_group("module for byzantine aggregate")
    byzantine_aggregate.add_argument("--byzantine_aggregate", action="store_true", help="")
    byzantine_aggregate.add_argument("--by_attack", type=str, help="None")
    byzantine_aggregate.add_argument("--by_defense", type=str, help="Avg")
    byzantine_aggregate.add_argument("--all_grads", action="store_true", help="")
    byzantine_aggregate.add_argument("--num_workers", type=int, help=1)
    byzantine_aggregate.add_argument("--by_workers", type=int, help=0)

    backdoor = parser.add_argument_group('module for backdoor defense module')
    backdoor.add_argument('--backdoor-test', action='store_true', help='enable backdoor testing for clients')
    backdoor.add_argument('--backdoor-test-frequency', type=int, default=10, help='frequency of malicious clients')
    backdoor.add_argument('--backdoor-test-boost', type=float, default=1.0, help='boost factor')
    backdoor.add_argument('--backdoor-defense', action='store_true', help='enable backdoor defense for server')
    backdoor.add_argument('--backdoor-defense-shrink', type=float, default=1, help='shrink threshold')
    backdoor.add_argument('--backdoor-defense-noise', type=float, default=0, help='standard deviation of noise added')

    return parser


def load_data(args, dataset_name, use_transform_policy):
    implemented = False  # only set to True if dataset's "use_transform_policy" is implemented
    transform_policy = get_policy(dataset_name, args.model, args.topn, args.augid) if use_transform_policy else None

    mnist_dataset = './../../../data/mnist'

    if dataset_name == 'mnistRGB':
        implemented = True
        # reshape and resize mnist to 3x32x32
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_mnistRGB(args.batch_size, args.partition_method, args.partition_alpha,
                                                 args.client_num_in_total, mnist_dataset, transform_policy,
                                                 apply_policy_to_validset=False)
    elif "mnist" in dataset_name:
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_mnist(args.batch_size, mnist_dataset + '/train', mnist_dataset + '/test')
        args.client_num_in_total = client_num

    else:
        implemented = True
        data_loader = load_partition_data_cifar100 if dataset_name == "cifar100" else load_partition_data_cifar10

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, transform_policy)

    if use_transform_policy and not implemented:
        raise NotImplementedError("use_transform_policy is True, but is not implemented for {}".format(dataset_name))

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def create_model(args, model_name, output_dim):
    if model_name == "lr" and "mnist" in args.dataset:
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "cnn" and "mnist" in args.dataset:
        model = CNN_OriginalFedAvg(True)
    elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
        model = resnet18()
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    # TODO
    elif model_name == 'mobilenet_v3':
        model = MobileNetV3(model_mode='LARGE')
    elif model_name == 'convnet' and 'cifar' in args.dataset:
        model = ConvNet(width=64, num_channels=3, num_classes=output_dim)
    elif model_name == 'convnet' and args.dataset == 'mnist2':
        model = ConvNet(width=28, num_channels=1, num_classes=output_dim)
    elif model_name == 'convnet' and args.dataset == 'mnistRGB':
        model = ConvNet(width=32, num_channels=3, num_classes=output_dim)
    elif model_name == "lr" and "cifar" in args.dataset:
        model = LogisticRegression(3 * 32 * 32, output_dim)
    elif model_name == "convnet_test" and 'mnist' in args.dataset:
        model = SampleConvNet()
    else:
        raise NotImplementedError
    return model


if __name__ == "__main__":

    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    if sys.platform == 'darwin':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # parse python script input parameters
    args = add_args(argparse.ArgumentParser()).parse_args()

    # customize the process name
    str_process_name = "FedAvg (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format=str(process_id) + ' - %(asctime)s %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        force=True)
    logging.basicConfig(level=logging.DEBUG,
                        format=str(process_id) + ' - %(asctime)s %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        force=True)
    hostname = socket.gethostname()

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if args.wandb_name == '':
        wandbname = f"FedAVG(d)-{args.model}-{args.dataset}-r{args.comm_round}-e{args.epochs}-lr{args.lr}"
        wandbname += "-data_recovery_protection" if args.data_recovery_protection else ""
        wandbname += "-differential_privacy" if args.differential_privacy else ""
        wandbname += "-byzantine_aggregate" + f' {args.by_defense} {args.by_attack} {args.num_workers} {args.by_workers} {args.all_grads}' if args.byzantine_aggregate else ""
        wandbname += "-backdoor-defense" if args.backdoor_defense else ""
        args.wandb_name = wandbname
    if process_id == 0 and not args.wandb_off:
        # https://wandb.ai/barryzzj/spfl
        # wandb.login(key='3f22eb6f0764f7f60db2e1d8b4c95e88248cfefd')
        # wandb.init(project='spfl', entity='barryzzj', name=args.wandb_name, config=args)
        wandb.login(key='7e1168bbcc86004163fff8fb46ac06b9dea62195')
        wandb.init(project='SPFL_TEST', entity='du22jk', name=args.wandb_name, config=args)

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Please check "GPU_MAPPING.md" to see how to define the topology
    device = mapping_processes_to_gpu_device_from_yaml_file(process_id, worker_number, args.gpu_mapping_file,
                                                            args.gpu_mapping_key)

    # load data
    dataset = load_data(args, args.dataset, args.data_recovery_protection)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

    # create model.
    model = create_model(args, model_name=args.model, output_dim=dataset[7])

    try:
        FedML_FedAvg_distributed(process_id, worker_number, device, comm,
                                 model, train_data_num, train_data_global, test_data_global,
                                 train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args)
    except Exception as e:
        print(e)
        logging.debug('traceback.format_exc():\n%s' % traceback.format_exc())
        MPI.COMM_WORLD.Abort()
