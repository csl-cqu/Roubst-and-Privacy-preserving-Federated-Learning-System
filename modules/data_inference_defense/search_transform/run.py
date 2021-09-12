import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from spfl_api.utils.pathfinder import *
from modules.data_inference_defense.search_transform.utils import aug_list_gen_path

parser = argparse.ArgumentParser(description='Run the whole procedure of transform policy search')
parser.add_argument('--skip_to_step', default=1, type=int, help='Skip to step i (start with 1)')
parser.add_argument('--model', default="ConvNet", type=str, help='Pretrained model.') #ResNet20-4
parser.add_argument('--dataset', default="cifar100", type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=100, type=int, help='Pretrained model\'s epoch (must divisible by 10.')
parser.add_argument('--gpunum', type=int, default=1, help='Gpu number')
parser.add_argument('--cmax', type=int, default=1600, help='Number of policies generate')
parser.add_argument('--processes', default=2, type=int, help="total processes, at least 2.")
parser.add_argument('-j', '--jobs', type=int, default=1, help='Allow N jobs at once. If error, try decrease this value')
args = parser.parse_args()


if __name__ == '__main__':
    pass
    # 1. gen_pretrained_model
    # 2. gen_aug_list
    # 3. calc_policy_score
    # 4. search_best_policy
    files = ['gen_pretrained_model.py',
             'gen_aug_list.py',
             'calc_policy_score.py',
             'search_best_policy.py']

    base_dir = os.path.join(MODULE_DATA_RECOV, 'search_transform')
    if args.skip_to_step <= 1:
        print("step 1. generate pretrained model...")
        file_abs = os.path.join(base_dir, files[0])
        if os.system(f"python3 {file_abs} --model {args.model} --dataset {args.dataset} --epochs {args.epochs} --processes {args.processes}") != 0:
            exit(-1)

    if args.skip_to_step <= 2:
        print("step 2. generate aug list...")
        file_abs = os.path.join(base_dir, files[1])
        if os.system(f"python3 {file_abs} --model {args.model} --dataset {args.dataset} --epochs {args.epochs} --gpunum {args.gpunum} --cmax {args.cmax} -j {args.jobs}") != 0:
            exit(-1)

    if args.skip_to_step <= 3:
        print("step 3. calculate policy score")
        file_abs = aug_list_gen_path(args.dataset, args.model, args.epochs)
        # if sys.platform == 'win32':
        #     if os.system(file_abs) != 0:
        #         exit(-1)
        # else:
        if os.system(f"bash {file_abs}") != 0:
            exit(-1)

    if args.skip_to_step <= 4:
        print("step 4. search beast policy")
        file_abs = os.path.join(base_dir, files[3])
        if os.system(f"python3 {file_abs} --model {args.model} --dataset {args.dataset} --epochs {args.epochs}") != 0:
            exit(-1)



