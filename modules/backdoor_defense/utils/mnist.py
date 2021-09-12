import copy
import logging

import torch

from spfl_api.distributed.fedavg.FedAVGTrainer import FedAVGTrainer


def create_malicious_dataset(benign_dataset):
    def convert(x):
        t = copy.deepcopy(x)
        t[:, 28 * 2 + 1] = t[:, 28 * 2 + 2] = t[:, 28 * 2 + 3] = t.max()
        t[:, 28 * 1 + 2] = t[:, 28 * 2 + 2] = t[:, 28 * 3 + 2] = t.max()
        return t

    return [(convert(x), torch.tensor([7] * y.shape[0])) for x, y in benign_dataset]


class MaliciousTrainer(FedAVGTrainer):
    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                 train_data_num, device, args, model_trainer):
        super(MaliciousTrainer, self).__init__(client_index, train_data_local_dict, train_data_local_num_dict,
                                               test_data_local_dict, train_data_num, device, args, model_trainer)

        self.malicious_dataset = None
        self.malicious_trainer = copy.deepcopy(self.trainer)

        if args.client_optimizer == 'sgd':
            self.malicious_optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.malicious_trainer.model.parameters()),
                lr=args.lr)
        else:
            self.malicious_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.malicious_trainer.model.parameters()),
                lr=args.lr, weight_decay=args.wd, amsgrad=True)

        logging.info('Malicious trainer initialized')

    def update_model(self, weights):
        super(MaliciousTrainer, self).update_model(weights)

        self.malicious_trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        super(MaliciousTrainer, self).update_dataset(client_index)

        self.malicious_dataset = create_malicious_dataset(self.train_data_local)

    def train(self, round_idx=None):
        if self.client_index % self.args.backdoor_test_frequency == 0:
            logging.info('Actually performing malicious training for client ' + str(self.client_index))

            origin = self.malicious_trainer.get_model_params()

            self.malicious_trainer.train(self.malicious_dataset, self.device, self.args, self.malicious_optimizer,
                                         dataset_size=self.local_sample_number, client_idx=self.client_index)

            benign, _ = super(MaliciousTrainer, self).train(round_idx)

            weights = self.malicious_trainer.get_model_params()
            weights = {key: benign[key] + self.args.backdoor_test_boost * (val - origin[key]) for key, val in
                       weights.items()}

            return weights, self.local_sample_number
        else:
            return super(MaliciousTrainer, self).train(round_idx)
