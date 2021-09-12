#!/usr/bin/env bash
GPU_MAPPING_KEY="mapping_standalone_4"
CLIENT_NUM=10
WORKER_NUM=3
MODEL=convnet
DISTRIBUTION=homo
ROUND=50
EPOCH=1
BATCH_SIZE=128
LR=0.03
DATASET=mnistRGB
DATA_DIR="./../../../data/mnistRGB"
CLIENT_OPTIMIZER=sgd
BACKEND=MPI
CI=0

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM

cd ../spfl_experiments/distributed/fedavg || exit

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
  --ci $CI
