import copy
import logging
import random
from collections import OrderedDict
from typing import Tuple, List

import numpy as np
import torch
import wandb
from torch import Tensor
from torch.nn import Parameter

from modules.backdoor_defense.utils.mnist import create_malicious_dataset
from modules.byzantine_resilience.defense import Avg, Krum, get_model_grads_from_one_row_grads, Median, Bulyan, Trimmed_mean, Dnc
from modules.byzantine_resilience.attack import  ARG_tailored,  Gaussian
from spfl_core.optimizer import Adam, SGD
from .utils import transform_list_to_tensor, transform_params_list_to_tensor


class FedAVGAggregator(object):
    '''used by server'''
    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                 args, model_trainer):
        self.trainer = model_trainer

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.model_optimizerparams_dict = dict()
        self.model_grads_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

        self.model_paramnames = [name for name, _ in self.trainer.get_model_namedparams()]

        if self.args.backdoor_test:
            self.malicious_train_data = dict()
            for key, val in self.train_data_local_dict.items():
                self.malicious_train_data[key] = create_malicious_dataset(val)
            self.malicious_test_data = create_malicious_dataset(self.test_global)
            self.malicious_val_data = create_malicious_dataset(self.val_global)

    def get_global_model_params(self) -> OrderedDict:
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters: OrderedDict):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params: OrderedDict, sample_num):
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def get_global_model_optimizerparams_grads(self) -> Tuple[List[Parameter], List[Tensor]]:
        return self.trainer.get_model_optimizerparams_grads()

    def set_global_model_optimizerparams(self, params: List[Parameter]):
        self.trainer.set_model_by_names_params(self.model_paramnames, params)

    def add_local_trained_grads(self, index, model_grads: Tensor, sample_num):
        self.model_grads_dict[index] = model_grads
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def add_local_trained_optimizerparam_grads(self, index, model_optimizerparams: Parameter, model_grads: Tensor,
                                               sample_num):
        self.model_optimizerparams_dict[index] = model_optimizerparams
        self.model_grads_dict[index] = model_grads
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        if not self.args.use_gradient:
            return self.aggregate_weights()
        else:
            grad_locals = []
            training_num = 0

            for idx in range(self.worker_num):
                if self.args.is_mobile == 1:
                    self.model_grads_dict[idx] = transform_params_list_to_tensor(self.model_grads_dict[idx])
                    if self.args.byzantine_aggregate:
                        self.model_optimizerparams_dict[idx] = transform_params_list_to_tensor(
                            self.model_optimizerparams_dict[idx])
                grad_locals.append((self.sample_num_dict[idx], self.model_grads_dict[idx]))
                training_num += self.sample_num_dict[idx]

            # if self.args.byzantine_aggregate:
            #     param_locals = [self.model_optimizerparams_dict[idx] for idx in range(self.worker_num)]


            # update the global model which is cached at the server side
            param_global, _ = self.get_global_model_optimizerparams_grads()

            if self.args.byzantine_aggregate:
                averaged_grads = self.by_module([l[1] for l in grad_locals],self.args, param_global, num_workers=self.args.num_workers, nbbzworkers=self.args.by_workers)
            else:
                averaged_grads = self.aggregate_grads(grad_locals, training_num)


            # server optimizer step to get w_global
            if self.args.client_optimizer == "sgd":
                server_optimizer = SGD(param_global, lr=1.0, momentum=0.99)
            else:
                server_optimizer = Adam(param_global, lr=self.args.lr, weight_decay=self.args.wd, amsgrad=True)
            # TODO transfer param_global to gpu?
            server_optimizer.step(averaged_grads)
            # set new params for global model
            self.set_global_model_optimizerparams(param_global)

            return self.get_global_model_params()





    def by_module(self,worker_paramters ,args, param_global, num_workers=1, nbbzworkers=0, device=torch.device('cpu')):


        if args.by_attack =='Gaussian':
            worker_grads = Gaussian(nbbzworkers, device).get_mal_update(worker_paramters, all_grads=args.all_grads)
        elif args.by_attack == 'ARG_tal-bulyan'or args.by_attack == 'ARG_tal-mkrum':
            worker_grads = ARG_tailored(nbbzworkers, device).get_mal_update(worker_paramters, 'bulyan', dev_type='sign', all_grads=args.all_grads)
        else:
            raise Exception('Incorrect Attack Type!')

        if args.by_defense == 'Avg':
            aggregate_grad = get_model_grads_from_one_row_grads(Avg().aggregate(worker_grads), param_global)
        elif args.by_defense == 'Krum':
            krum_grads, indices =  Krum(num_workers, nbbzworkers, 1).aggregate(worker_grads)
            aggregate_grad = get_model_grads_from_one_row_grads(krum_grads, param_global)
        elif args.by_defense == 'MKrum':
            krum_grads, indices =  Krum(num_workers, nbbzworkers, num_workers-2*nbbzworkers).aggregate(worker_grads)
            aggregate_grad = get_model_grads_from_one_row_grads(krum_grads, param_global)
        elif args.by_defense == 'Median':
            aggregate_grad = get_model_grads_from_one_row_grads(Median(num_workers, nbbzworkers).aggregate(worker_grads), param_global)
        elif args.by_defense == 'Bulyan':
            aggregate_grad = get_model_grads_from_one_row_grads(Bulyan(num_workers, nbbzworkers).aggregate(worker_grads), param_global)
        elif args.by_defense == 'Trimmed-mean':
            aggregate_grad = get_model_grads_from_one_row_grads(Trimmed_mean(num_workers,nbbzworkers).aggregate(worker_grads), param_global)
        elif args.by_defense == 'Dnc':
            aggregate_grad = get_model_grads_from_one_row_grads(Dnc(nbbzworkers,1).aggregate(worker_grads)[0], param_global)
        else:
            raise Exception('Incorrect Defense Type!')


        return aggregate_grad

    def aggregate_weights(self):
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        return averaged_params

    def aggregate_grads(self, model_grads_list, training_num):
        (num0, averaged_grads) = model_grads_list[0]
        for k in range(len(averaged_grads)):
            for i in range(0, len(model_grads_list)):
                local_sample_number, local_model_grads = model_grads_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_grads[k] = local_model_grads[k] * w
                else:
                    averaged_grads[k] += local_model_grads[k] * w

        return averaged_grads

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx, prefix='Std.'):
        if self.trainer.test_on_the_server(
                self.train_data_local_dict, self.test_data_local_dict, self.device, self.args):
            return

        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info('No. ' + str(round_idx))
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []
            for client_idx in range(self.args.client_num_in_total):
                # train data
                metrics = self.trainer.test(self.train_data_local_dict[client_idx], self.device, self.args)
                train_tot_correct, train_num_sample, train_loss = \
                    metrics['test_correct'], metrics['test_total'], metrics['test_loss']
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))

                if self.args.ci == 1:
                    break

            # test on training dataset
            train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            train_loss = sum(train_losses) / sum(train_num_samples)
            if not self.args.wandb_off:
                wandb.log({f'{prefix} train acc': train_acc, "round": round_idx})
                wandb.log({f'{prefix} train loss': train_loss, "round": round_idx})
            logging.info(prefix + ' train acc/loss=%.2f%%/%f' % (train_acc * 100, train_loss))

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)

            test_tot_correct, test_num_sample, test_loss = \
                metrics['test_correct'], metrics['test_total'], metrics['test_loss']
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            if not self.args.wandb_off:
                wandb.log({f'{prefix} test  acc': test_acc, "round": round_idx})
                wandb.log({f'{prefix} test  loss': test_loss, "round": round_idx})
            logging.info(prefix + ' test  acc/loss=%.2f%%/%f' % (test_acc * 100, test_loss))
