
export CUDA_VISIBLE_DEVICES=0
dataset=cifar100
epoch=50
model=convnet

#--use_pretrained_model 

python3 ./attack.py --dataset=$dataset --model=$model --mode=normal --optim=inversed --epochs=$epoch #--hide_output_img

for aug_list in '41-11-31' '34-11-36' '11-41-28' '11-28-49' '11-39-8';
do
{
echo $aug_list
python3 -u ./attack.py --dataset=$dataset --model=$model --mode=policy --optim=inversed --epochs=$epoch --aug_list=$aug_list #--hide_output_img
}
done
wait
