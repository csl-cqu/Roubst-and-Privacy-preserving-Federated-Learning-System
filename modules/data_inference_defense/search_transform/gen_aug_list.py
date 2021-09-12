# python -u searchalg/gen_aug_list.py  --model=ResNet20-4 --dataset=cifar100
import argparse
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from spfl_api.utils.pathfinder import *
from modules.data_inference_defense.search_transform.utils import aug_list_gen_path

seed=23333
# random.seed(seed)

parser = argparse.ArgumentParser(description='Random sample list of augmentation indices')
parser.add_argument('--model', default="convnet", type=str, help='Pretrained model.') #ResNet20-4
parser.add_argument('--dataset', default="cifar100", type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=100, type=int, help="Pretrained model's epochs")
parser.add_argument('--gpunum', type=int, default=1, help='Gpu number')
parser.add_argument('--cmax', type=int, default=1600, help='Number of policies generate')
parser.add_argument('-j', '--jobs', type=int, default=1, help='Allow N jobs at once. If error, try decrease this value')
args = parser.parse_args()


# num_per_gpu = 1
gpunum = args.gpunum
cmax = args.cmax
jobs = args.jobs

def write_sh(aug_lists):
    output = aug_list_gen_path(args.dataset, args.model, args.epochs)
    f = open(output, "w", newline='\n')
    for i in range(jobs):
        lines = []
        lines.append('{')
        for idx in range(i * len(aug_lists) // jobs, (i+1) * len(aug_lists) // jobs):
            aug_list = [str(sch) for sch in aug_lists[idx]]
            suf = '-'.join(aug_list)
            path = os.path.join(MODULE_DATA_RECOV, "search_transform", "calc_policy_score.py").replace('\\','/')
            cmd = 'CUDA_VISIBLE_DEVICES={} python3 {} --aug_list={} --model={} --dataset={} --epochs={}\nif [ $? != 0 ]; then exit; fi'.format(
                i % gpunum, path, suf, args.model, args.dataset, args.epochs)
            lines.append(cmd)
        lines.append('}&')
        f.write('\n'.join(lines))
        f.write('\n')
    f.write('wait')
    f.close()

def main():
    # aug_list_path = aug_list_gen_path(args.dataset, args.model, args.epochs)
    # sys.stdout = open(aug_list_path, "w")
    init_aug_lists = {
        'convnet': [[21,13,3], [7,4,15]],
        'ResNet20-4': [[3,1,7], [43,18,18], [3,18,28], [7,3]],
        'ResNet20': [[3,1,7], [43,18,18], [3,18,28], [7,3]],
    }
    aug_lists = init_aug_lists[args.model] if args.model in init_aug_lists.keys() else list()
    for _ in range(cmax-len(aug_lists)):
        aug_list = list()
        for i in range(3):
            aug_list.append(random.randint(-1, 49))
        aug_list = [i for i in aug_list if i != -1]
        aug_lists.append(aug_list)

    write_sh(aug_lists)
    # sys.stdout.flush()

if __name__ == '__main__':
    main()
