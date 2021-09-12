import numpy as np
import torch

# from defense import Krum, Trimmed_mean, Dnc, Median
from .defense import Krum, Trimmed_mean, Dnc, Median
"""
Objective
Capability
Knowledge
"""


def get_grads_from_params(model_params_list):
    """
    get gradients from model.parameters()
    view all gradients of one node to one-dimensional
    number of node: n, dim_size: d
    return a tensor which shape is [n, d]
    """
    worker_grads = []
    for model_params in model_params_list:
        worker_grad = []
        for param in model_params:
            worker_grad = param.data.view(-1) if not len(worker_grad) else torch.cat((worker_grad, param.view(-1)))

        worker_grads = worker_grad[None, :] if not len(worker_grads) else torch.cat(
            (worker_grads, worker_grad[None, :]), dim=0)

    return worker_grads


class ARG_tailored():
    def __init__(self, nbbzworkers, device):
        assert nbbzworkers >= 0, 'number of byzantine workers must greater than zero'
        self.nbbzworkers = nbbzworkers
        self.device = device

    def get_mal_update(self, worker_grads, ARG, dev_type='inv_unit', all_grads=True, th=5.0):
        """
        Args:
            worker_grad: gradients of workers
            dev_type: type of deviation
            all_grads: do the attackers know benign worker_grads
        Return:
            mal_grads: malicious tensor, shape [n,d]
        """

        if isinstance(worker_grads, list):
            worker_grads = get_grads_from_params(worker_grads)

   
        worker_grads_back = worker_grads
        if not all_grads:
            worker_grads = worker_grads[:self.nbbzworkers]

        grads_mean = torch.mean(worker_grads, dim=0)
        #choose different deviation
        if dev_type == 'inv_unit':
            deviation = grads_mean / torch.norm(grads_mean)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(grads_mean)
        elif dev_type == 'std':
            deviation = torch.std(worker_grads, 0)

        if ARG == 'trimmed_mean':
            mal_update = self.tailored_trmean(worker_grads, grads_mean, self.nbbzworkers, deviation, all_grads)
        elif ARG == 'median':
            mal_update = self.tailored_median(worker_grads, grads_mean, self.nbbzworkers, deviation, all_grads,
                                                     threshold=th)
        elif ARG == 'krum' or ARG == 'mkrum':
            mal_update = self.tailored_mkrum(worker_grads, grads_mean, self.nbbzworkers, deviation, all_grads)
        elif ARG == 'bulyan':
            mal_update =  self.tailored_bulyan(worker_grads, grads_mean, self.nbbzworkers, deviation, all_grads)
        elif ARG == 'dnc':
            mal_update =  self.tailored_dnc(worker_grads, all_grads)
            mal_updates = torch.stack([mal_update]*self.nbbzworkers)
        else:
            raise Exception('No correct ARG: one of the trimmed_mean, median, krum and bulyan')

        mal_updates = torch.stack([mal_update] * self.nbbzworkers)
        mal_updates = torch.cat((mal_updates, worker_grads_back[self.nbbzworkers:]), 0)
        return mal_updates

    def tailored_mkrum(self, all_updates, model_re, n_attackers, deviation, all_grads, threshold=20.0):

        lamda = torch.Tensor([threshold]).cuda(self.device)  # compute_lambda_our(all_updates, model_re, n_attackers)
        n, d = all_updates.shape
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)

            if all_grads:
                mal_updates = torch.stack([mal_update] * n_attackers)
                # mal_updates = torch.cat((mal_updates, all_updates), 0)
                mal_updates = torch.cat((mal_updates, all_updates[n_attackers:]), 0)
            else:
                mal_updates = torch.cat((mal_update[None, :], all_updates), 0)
                n_attackers = 1

           
            # set multi_m = 1, to satisfy below judgement condition
            # agg_grads, krum_candidate = Krum(n, n_attackers, multi_m=n-2*n_attackers-2).aggregate(mal_updates)
            agg_grads, krum_candidate = Krum(n, n_attackers, multi_m=1).aggregate(mal_updates)

            #lamda success when aggregate method choose malicious gradient
            if any(krum_candidate < n_attackers):
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2
        mal_update = (model_re - lamda_succ * deviation)

        return mal_update

    def tailored_bulyan(self, all_updates, model_re, n_attackers, deviation, all_grads, threshold=20.0):

        # lamda = torch.Tensor([threshold]).cuda(self.device)  # compute_lambda_our(all_updates, model_re, n_attackers)
        lamda = torch.Tensor([threshold]).to(self.device)
        n, d = all_updates.shape
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            if all_grads:
                mal_updates = torch.stack([mal_update] * n_attackers)
                # mal_updates = torch.cat((mal_updates, all_updates), 0)
                mal_updates = torch.cat((mal_updates, all_updates[n_attackers:]), 0)
            else:
                mal_updates = torch.cat((mal_update[None, :], all_updates), 0)
                n_attackers = 1

            # set multi_m = n-2*n_attackers-2, to satisfy below judgement condition
            agg_grads, krum_candidate = Krum(n, n_attackers, multi_m=n - 2 * n_attackers).aggregate(mal_updates)

            if torch.sum(krum_candidate < n_attackers) == n_attackers:

                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2
        mal_update = (model_re - lamda_succ * deviation)

        return mal_update

    def tailored_dnc(self, all_updates, model_re, n_attackers, deviation, all_grads, threshold=20.0):

        lamda = torch.Tensor([threshold]).cuda(self.device) #compute_lambda_our(all_updates, model_re, n_attackers)
        n,d = all_updates.shape
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            if all_grads:
                mal_updates = torch.stack([mal_update] * n_attackers)
                # mal_updates = torch.cat((mal_updates, all_updates), 0)
                mal_updates = torch.cat((mal_updates, all_updates[n_attackers:]), 0)
            else:
                mal_updates = torch.cat((mal_update[None, :], all_updates), 0)
                n_attackers = 1

            agg_grads, krum_candidate = Dnc(n_attackers, 1).aggregate(mal_updates)

            
            if torch.sum(krum_candidate < n_attackers) == n_attackers:
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2
        mal_update = (model_re - lamda_succ * deviation)

        return mal_update

    def tailored_trmean(self, all_updates, model_re, n_attackers, deviation, all_grads, threshold=5.0, threshold_diff=1e-5):
        
        
        lamda = torch.Tensor([threshold]).cuda(self.device)  # compute_lambda_our(all_updates, model_re, n_attackers)

        threshold_diff = threshold_diff
        prev_loss = -1
        lamda_fail = lamda
        lamda_succ = 0

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            if all_grads:
                mal_updates = torch.stack([mal_update] * n_attackers)
                # mal_updates = torch.cat((mal_updates, all_updates), 0)
                mal_updates = torch.cat((mal_updates, all_updates[n_attackers:]), 0)
            else:
                mal_updates = torch.cat((mal_update[None, :], all_updates), 0)
                n_attackers = 1

            agg_grads = Trimmed_mean(all_updates.shape[0], n_attackers).aggregate(mal_updates)

            loss = torch.norm(agg_grads - model_re)

            if prev_loss < loss:
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2
            prev_loss = loss

        mal_update = (model_re - lamda_succ * deviation)

        return mal_update

    def tailored_median(self, all_updates, model_re, n_attackers, deviation, all_grads, threshold=5.0,
                               threshold_diff=1e-5):

        # lamda = torch.Tensor([threshold]).cuda(self.device)  # compute_lambda_our(all_updates, model_re, n_attackers)
        lamda = torch.Tensor([threshold]).to(self.device)
        threshold_diff = threshold_diff
        prev_loss = -1
        lamda_fail = lamda
        lamda_succ = 0

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            if all_grads:
                mal_updates = torch.stack([mal_update] * n_attackers)
                # mal_updates = torch.cat((mal_updates, all_updates), 0)
                mal_updates = torch.cat((mal_updates, all_updates[n_attackers:]), 0)
            else:
                mal_updates = torch.cat((mal_update[None, :], all_updates), 0)
                n_attackers = 1

            agg_grads = Median(all_updates.shape[0], n_attackers).aggregate(mal_updates)

            loss = torch.norm(agg_grads - model_re)

            if prev_loss < loss:
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2
            prev_loss = loss

        mal_update = (model_re - lamda_succ * deviation)

        return mal_update


class Gaussian():
    def __init__(self, nbbzworkers, device):
        assert nbbzworkers >= 0, 'number of byzantine workers must greater than zero'
        self.nbbzworkers = nbbzworkers
        self.device = device

    def get_mal_update(self, worker_grads, all_grads=False):
        if isinstance(worker_grads, list):
            worker_grads = get_grads_from_params(worker_grads)

        n, d = worker_grads.shape
        rand = torch.from_numpy(np.random.uniform(0, 1, [self.nbbzworkers, d])).type(torch.FloatTensor).to(self.device)
        # rand = torch.from_numpy(np.random.uniform(0, 1, [self.nbbzworkers, d])).type(torch.FloatTensor)

        mal_updates = torch.cat((rand, worker_grads[self.nbbzworkers:]), dim=0)
        return mal_updates
