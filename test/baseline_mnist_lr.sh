#!/usr/bin/env bash

hostname >mpi_host_file

cd ../spfl_experiments/distributed/fedavg || exit

case $(hostname) in
"pub-MS-7B33") GPU_MAPPING="mapping_onehost_2_6"; WORKER_NUM=6 ;;
*            ) GPU_MAPPING="mapping_onehost_1_2"; WORKER_NUM=2 ;;
esac

CLIENT_NUM_PER_ROUND=$(expr $WORKER_NUM - 1)

mpirun -np $WORKER_NUM -hostfile ../../../test/mpi_host_file python3 ./main_fedavg.py \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key $GPU_MAPPING \
  --model lr \
  --dataset mnist \
  --data_dir "../../../data/MNIST" \
  --partition_method hetero \
  --client_num_in_total 10 \
  --client_num_per_round $CLIENT_NUM_PER_ROUND \
  --comm_round 200 \
  --epochs 1 \
  --client_optimizer sgd \
  --batch_size 10 \
  --lr 0.05 \
  --backend MPI \
  --ci 0 \
  --wandb-off\
  "$@"
