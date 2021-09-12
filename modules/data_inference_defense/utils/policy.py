import csv
import os
from typing import List

import numpy as np

from .transformation import transformations


class Policy:
    pass

class PolicySingle(Policy):
    def __init__(self, transform_list: List[int]):
        self.transform_list = transform_list

    def __call__(self, img):
        for transform_id in self.transform_list:
            img = transformations[transform_id](img)
        return img

class PolicyHybrid(Policy):
    def __init__(self, policy_list: List[List[int]]):
        self.policy_list = [PolicySingle(l) for l in policy_list]

    def __call__(self, img):
        idx = np.random.randint(0, len(self.policy_list))
        select_policy = self.policy_list[idx]
        img = select_policy(img)
        return img

def read_policy_list_from_csv(csv_pathname):
    path = os.path.abspath(csv_pathname)
    assert os.path.exists(path), "csv file not found!"
    policy_dict = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if v:
                    # change "1-2-3" into [1, 2, 3]
                    policy = list(map(int, v.split("-")))
                    policy_dict.setdefault(k, []).append(policy)
    return policy_dict

def make_policy_single(str:str):
    return PolicySingle(list(map(int, str.split('-'))))

def make_policy_hybrid(policystr_list: List[str]):
    policy_list = [list(map(int, str.split('-'))) for str in policystr_list]
    return PolicyHybrid(policy_list)
