import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import argparse

from spfl_api.utils.pathfinder import *
from modules.data_inference_defense.search_transform.utils import pretrained_model_path

parser = argparse.ArgumentParser(description='Generate pretrained model for calculating privacy score.')
parser.add_argument('--model', default="ConvNet", type=str, help='Vision model.') # ConvNet
parser.add_argument('--dataset', default="mnist_original", type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=100, type=int, help="Pretrained model's epochs")
parser.add_argument('--processes', default=2, type=int, help="total processes, at least 2.")
args = parser.parse_args()

assert args.processes >= 2, "need at least 2 processes to train model."

# def load_checkpoint_if_exists(model, save_dir):
#     files = [int(filename[:-4]) for filename in os.listdir(save_dir) if filename.endswith('.pth')]
#     if files:
#         max_file = f"{max(files)}.pth"
#         print("load " + max_file)
#         model.load_state_dict(torch.load(os.path.join(save_dir, max_file)))
#         return max(files)
#     return 0
# Ms is used for privacy quantification. It is trained only with 10% of the original training set for 50 epochs.
def main():
    model_path = pretrained_model_path(args.dataset, args.model, args.epochs)
    save_dir = os.path.dirname(model_path)

    if os.path.exists(model_path):
        print("pretrained model exists")
        return 0

    # use toolbox to pretrain model
    file_abs = MAIN_FEDAVG
    sh_pathname = "./gen_pretrained_model.sh"
    sh = f'''#!/usr/bin/env bash

GPU_MAPPING_KEY="mapping_standalone_{args.processes}"
CLIENT_NUM=2
WORKER_NUM={args.processes-1}
MODEL={args.model}
DISTRIBUTION=homo
ROUND={args.epochs}
EPOCH=1
BATCH_SIZE=128
LR=0.03
DATASET={args.dataset}
DATA_DIR="./../../../data/{args.dataset}"
SAVE_PATH={model_path}
CLIENT_OPTIMIZER=sgd
BACKEND=MPI
CI=0

PROCESS_NUM=`expr $WORKER_NUM + 1`

cd ../../../spfl_experiments/distributed/fedavg || exit

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedavg.py \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key $GPU_MAPPING_KEY \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --partition_method $DISTRIBUTION  \
  --client_num_in_total $CLIENT_NUM \
  --client_num_per_round $WORKER_NUM \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --client_optimizer $CLIENT_OPTIMIZER \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --backend $BACKEND \
  --ci $CI \
  --save_path $SAVE_PATH \
  --wandb-off'''

    with open(sh_pathname, 'w') as f:
        f.write(sh)

    return os.system(f"sh {sh_pathname}")

if __name__ == '__main__':
    if main() != 0:
        exit(-1)
