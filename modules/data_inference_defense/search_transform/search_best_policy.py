import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import argparse

from modules.data_inference_defense.search_transform.utils import *

parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--model', default="convnet", type=str, help='Pretrained model.')
parser.add_argument('--dataset', default="cifar100", type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=100, type=int, help="Pretrained model's epochs")
args = parser.parse_args()


if __name__ == '__main__':
    pri_root = pri_score_path(args.dataset, args.model, args.epochs, assert_path=True)
    acc_root = acc_score_path(args.dataset, args.model, args.epochs, assert_path=True)
    # maxpath, maxval = None, -sys.maxsize
    # minpath, minval = None, sys.maxsize
    ACC_THRESH = -85

    results = list()
    for policy_pathname in os.listdir(acc_root):
        aug_list = policy_pathname[:-4]
        acc_pathname = os.path.join(acc_root, policy_pathname)
        acc_score_list = np.load(acc_pathname).tolist()
        avg_acc_score = np.mean(acc_score_list)
        if avg_acc_score >= ACC_THRESH:

            pri_pathname = os.path.join(pri_root, policy_pathname)
            pri_score_list = np.load(pri_pathname).tolist()
            avg_pri_score = np.mean(pri_score_list)

            results.append((aug_list, avg_pri_score, avg_acc_score))
        else:
            print("discard {}, acc score {}".format(policy_pathname, avg_acc_score))

    result_dir = result_path(args.dataset, args.model, args.epochs)
    f = open(result_dir, "w", newline='\n')
    f.write("policy,avg_pri_score,avg_acc_score\n")
    # sort by pri score
    results.sort(key=lambda x: x[1])
    f.writelines([f"{result[0]},{result[1]},{result[2]}\n" for result in results])
    f.close()
    # print(results)
    # for idx, result in enumerate(results):
    #     print(result)
    #     f.write(result)
    #     f.write('\n')
