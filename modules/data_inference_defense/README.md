## Introduction

data_inference_defense module includes three parts: attack, defense, automatic search algorithm

**defense** contains the searched policies for each dataset, defined in .csv file.

**attack** tries to recover input dataset and calculates average psnr value.

**search_transform** is the automatic search algorithm for the best transform policies.



## Usage

## To search best transform policies for preprocessing
1. `cd search_transform`
2. `python3 run.py --model MODEL_NAME --dataset DATASET --epochs 50 --gpunum GPUS -j 2`

   - Model name can be found in `inversefed\nn\models.py`.

   - Dataset only implemented cifar10 and cifar100.

3. You can find results in `search_transform/search_result`. Copy first N policies and put them in `data_preprocessing/policy_list_DATASET.csv`, the csv file looks like:

```csv
resnet20,convnet
3-1-7,21-13-3
43-18-18,7-4-15
```

## Usage in toolbox

1. add `--data_recovery_protection` to `main_fedavg.py`'s arg

2. specify policy used: `--augid N`. Where N is the index in `modules/data_recovery_defense/data_preprocessing/policy_list_{dataset}.csv`

3. use hybrid policies: `--topn N`. Where top N hybrid policies in `modules/data_recovery_defense/data_preprocessing/policy_list_{dataset}.csv` are used.

## To validate defense ability against attack (calculate PSNR value)

1. `cd data_recovery_validation`
2. make a `.sh` file looks like:

```sh
export CUDA_VISIBLE_DEVICES=0
dataset=mnistRGB
epoch=50
model=convnet

python3 ./validation.py --dataset=$dataset --model=$model --mode=normal --optim=inversed --use_pretrained_model --epochs=$epoch --hide_output_img

for aug_list in '21-13-3+7-4-15' '21-13-3' '7-4-15';
do
{
echo $aug_list
python3 -u ./validation.py --dataset=$dataset --model=$model --mode=policy --optim=inversed --use_pretrained_model --epochs=$epoch --aug_list=$aug_list  --hide_output_img
}
done
wait
```

change `dataset`, `epoch`, `model` and `aug_list` to your configuration. `+` in `aug_list` means hybrid these policies.

3. run the `.sh` file, output will be in `data_recovery_validation/output`



# References

>[1] Zhu, Ligeng, and Song Han. "Deep leakage from gradients." *Federated learning*. Springer, Cham, 2020. 17-31.
>
>[2] Gao, Wei, et al. "Privacy-preserving collaborative learning with automatic transformation search." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2021.

