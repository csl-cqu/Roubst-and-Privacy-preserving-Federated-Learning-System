from functools import reduce

import numpy as np
import torch


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


def get_model_grads_from_one_row_grads(grads, model_parameters):
    """
    transform one row grads to model parameters shape grads
    when using this method for optimizer to step, 
    pay attention to move each grad in model_grads to correct device
    """
    model_grads = []
    idx = 0
    for i, param in enumerate(model_parameters):
        grad = grads[idx:idx + len(param.data.view(-1))].reshape(param.data.shape)
        idx += len(param.data.view(-1))
        model_grads.append(grad)

    return model_grads


class Krum():
    def __init__(self, nbworkers, nbbzworkers, multi_m):

        # '''
        # Args:
        #     model_params_list: model params list
        #     nbworkers: number of workers
        #     nbbzworkers: number of byzantine workers
        #     multi_m: the parameter m in Krum algorithm
        # '''

        self.nbworkers = nbworkers
        self.nbbzworkers = nbbzworkers
        self.multi_m = multi_m
        #  n - f - 2
        self.nbselected = self.nbworkers - self.nbbzworkers - 2

    def aggregate(self, model_params_list):
        assert len(model_params_list) > 0, 'Empty model parameters to aggregate!'
        return self._aggregate(model_params_list)

    def _aggregate(self, model_params_list):

        if isinstance(model_params_list, list):
            worker_grads = get_grads_from_params(model_params_list)
        else:
            worker_grads = model_params_list

        # if nbselected equals nbbzworkers, use average method
        if self.nbselected == self.nbbzworkers:
            return torch.mean(worker_grads, dim=0)

        else:

            distances = []
            for i, worker_grad in enumerate(worker_grads):
                # calculate distance between i and others
                # remove current grad from worker_grads
                remove = torch.cat((worker_grads[0:i], worker_grads[i + 1:]), dim=0)
                distance = torch.norm((remove - worker_grad), dim=1) ** 2  # distance shape [n-1,]

                distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]),
                                                                                   dim=0)  # cat tensor along row
            # distances shape [n, n-1]
            # sort distance along column
            distances = torch.sort(distances, dim=1)[0]
            #calculate sorce for each grads, sum nbselected distances
            scores = torch.sum(distances[:, :self.nbselected], dim=1) #socres shape [n,1]
            #find the indices of m smallest socre
            # print('before selected indices\n', torch.argsort(scores))
            indices = torch.argsort(scores)[:self.multi_m]
            return torch.mean(worker_grads[indices], dim=0), indices


class Avg():

    def aggregate(self, model_params_list):
        assert len(model_params_list) > 0, 'Empty model parameters to aggregate!'
        return self._aggregate(model_params_list)

    def _aggregate(self, model_params_list):
        # '''
        # Args:
        #     model_params_list: model params list
        # Returns
        #     avg_params: average number of model dict
        # '''

        if isinstance(model_params_list, list):
            worker_grads = get_grads_from_params(model_params_list)
        else:
            worker_grads = model_params_list

        # worker_grads = get_grads_from_params(model_params_list)
        return torch.mean(worker_grads, dim=0)
       
class Bulyan():
    def __init__(self, nbworkers, nbbzworkers):

        assert nbworkers > 2 * nbbzworkers, 'nubmer of workers must more than the double of number of byzantine workers'
        self.nbworkers = nbworkers
        self.nbbzworkers = nbbzworkers
        self.nbselected = self.nbworkers - self.nbbzworkers - 2

    def aggregate(self, model_params_list):
        assert len(model_params_list) > 0, 'Empty list of parameters to aggregate'
        return self._aggregate(model_params_list)

    def _aggregate(self, model_params_list):

        if isinstance(model_params_list, list):
            worker_grads = get_grads_from_params(model_params_list)
        else:
            worker_grads = model_params_list


        #using krum choose first n-2f parameters

        bulyan_set = [] # n - 2f gradients
        candidate_poor = [] # index of n-2f gradients in total gradients
        total_poor = np.arange(self.nbworkers)

        distances = []
        for i, worker_grad in enumerate(worker_grads):
            # calculate distance for every grad
            # remove current grad from worker_grads
            remove = torch.cat((worker_grads[:i], worker_grads[i + 1:]), dim=0)
            distance = torch.norm((remove - worker_grad), dim=1) ** 2  # distance shape [n-1,]

            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]),
                                                                               dim=0)  # cat tensor along row
        # distances shape [n, n-1]
        # sort distance along column
        distances = torch.sort(distances, dim=1)[0]
        # calculate sorce for each grads, sum nbselected distances
        scores = torch.sum(distances[:, :self.nbselected], dim=1)  # socres shape [n,1]
        # find the indices of n-f-2 smallest socre
        indices = torch.argsort(scores)[:self.nbworkers - 2 * self.nbbzworkers]
        # append the smallest gradient index to candidate poor
        # candidate_poor.append(total_poor[indices[0].cpu().numpy()])
        # #remove append index
        # total_poor = np.delete(total_poor, indices[0].cpu().numpy())
        # #append choose gradient to bulyan_set
        # bulyan_set = worker_grads[indices[0]][None, :] if not len(bulyan_set) else torch.cat((bulyan_set, worker_grads[indices[0]][None, :]), dim=0)

        bulyan_set = worker_grads[indices]
        # print('in bulyan indices', indices)

        # remove append gradient
        # worker_grads = torch.cat((worker_grads[:indices[0]], worker_grads[indices[0]+1:]), dim=0)

        theta, dim = bulyan_set.shape

        # compute median in every dimention, bulyan_set's shape [n-2f, dim]
        grads_median = torch.median(bulyan_set, dim=0)[0]  # shape [dim]
        # compute difference value between gradient and median in every dimention
        diff = torch.abs(bulyan_set - grads_median)
        # find the theta-2f cloest gradient indices
        cloest_indices = torch.argsort(diff, dim=0)[:theta - 2 * self.nbbzworkers]
        # get gradients using cloest_indices
        bulyan_gradents = bulyan_set[cloest_indices, torch.arange(dim)[None, :]]  # shape [n-4f, dim]
        # return mean of bulyan_gradients in every dimention
        return torch.mean(bulyan_gradents, dim=0)

class Dnc():
    def __init__(self, nbbzworkers=0, filter_frac=0):
        assert nbbzworkers >= 0 and filter_frac >= 0, 'number of nbbzworker and filter_frac must larger than zero'
        self.nbbzworkers = nbbzworkers
        self.filter_frac = filter_frac

    def aggregate(self, model_params_list):
        return self._aggregate(model_params_list)
    def _aggregate(self, model_params_list, iters=1):
        # worker_grads = get_grads_from_params(model_params_list)

        if isinstance(model_params_list, list):
            worker_grads = get_grads_from_params(model_params_list)
        else:
            worker_grads = model_params_list

        n, d = worker_grads.shape
        max_dim = 10000
        tao_good = []
        #calculate reserve gradient number
        nbselected = int(n - np.floor(self.filter_frac*self.nbbzworkers))
        for i in range(iters):
            #random select dimensions
            # indice = np.sort(np.random.choice(d, np.random.randint(1, max_dim), replace=False))
            indice = np.sort(np.random.choice(d, max_dim, replace=False))
            grads = worker_grads[:, indice]
            # compute mean
            grads_mean = torch.mean(grads, dim=0)
            # gcentered
            gcentered = (grads - grads_mean) / np.sqrt(n)  # shape [n, r]
            # compute top right singular eigenvector
            # print(gcentered.shape, d)
            _, _, v_t = torch.svd(gcentered) #v_t shape [r,n]
            #compute score using norm 
            scores = torch.norm(torch.mm(gcentered , v_t), dim=1)**2
            #select the smallest nbselected scores
            indice = torch.argsort(scores)[:nbselected]

            tao_good.append(indice)

        # find intersection in tao_good
        tao_good = (i.cpu().numpy() for i in tao_good)
        final_indice = reduce(np.intersect1d, tao_good)
        tao_final = worker_grads[final_indice, :]
        return torch.mean(tao_final, dim=0), final_indice


class Trimmed_mean():
    def __init__(self, nbworkers, nbbzworkers):

        assert nbworkers > 2 * nbbzworkers, 'nubmer of workers must more than the double of number of byzantine workers'
        self.nbworkers = nbworkers
        self.nbbzworkers = nbbzworkers

    def aggregate(self, model_params_list):
        assert len(model_params_list) > 0, 'Empty model parameters to aggregate!'
        return self._aggregate(model_params_list)

    def _aggregate(self, model_params_list):

        if isinstance(model_params_list, list):
            worker_grads = get_grads_from_params(model_params_list)
        else:
            worker_grads = model_params_list

        # beta = int(np.floor(self.nbbzworkers/2))
        beta = self.nbbzworkers
        sorted_grads = torch.sort(worker_grads, dim=0)[0]
        return torch.mean(sorted_grads[beta:self.nbworkers - beta, :], dim=0)


class Median():
    def __init__(self, nbworkers, nbbzworkers):

        assert nbworkers > 2 * nbbzworkers, 'nubmer of workers must more than the double of number of byzantine workers'
        self.nbworkers = nbworkers
        self.nbbzworkers = nbbzworkers

    def aggregate(self, model_params_list):
        assert len(model_params_list) > 0, 'Empty model parameters to aggregate!'
        return self._aggregate(model_params_list)

    def _aggregate(self, model_params_list):

        if isinstance(model_params_list, list):
            worker_grads = get_grads_from_params(model_params_list)
        else:
            worker_grads = model_params_list

        return torch.median(worker_grads, dim=0)[0]
