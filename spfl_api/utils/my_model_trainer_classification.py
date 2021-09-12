import copy
from collections import OrderedDict
from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import Parameter

from spfl_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def get_model_namedparams(self) -> Tuple[str, Parameter]:
        return self.model.cpu().named_parameters()

    def set_model_by_names_params(self, names: List[str], params: List[Parameter]):
        state_dict = copy.deepcopy(self.model.state_dict())
        assert len(names) == len(params)
        for i in range(len(params)):
            state_dict[names[i]] = params[i]
        self.model.load_state_dict(state_dict)

    def get_model_optimizerparams_grads(self) -> Tuple[List[Parameter], List[Tensor]]:
        grads = []
        params = []
        for p in self.model.cpu().parameters():
            grads.append(p.grad)
            params.append(p)
        return params, grads

    def get_model_params(self) -> OrderedDict:
        return self.model.cpu().state_dict()
        # return self.model.state_dict()

    def set_model_params(self, model_parameters: OrderedDict):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, optimizer, round_idx=None, dataset_size=None, client_idx=None):
        model = self.model
        model = model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.differential_privacy and self.dp_module.not_attached():
            assert dataset_size is not None
            assert self.dp_module is not None
            assert client_idx is not None
            self.dp_module.init_new_engine(model=model, optimizer=optimizer, device=device, client=client_idx,
                                           epochs=args.epochs, sample_size=dataset_size, batch_size=args.batch_size,
                                           sigma=args.dp_sigma, delta=args.dp_sigma, grad_norm=args.grad_norm)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels.long())
                loss.backward()

                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        if args.differential_privacy and self.id != -1:
            self.dp_module.update_all_spent(log=True, round_idx=round_idx)
        # if args.differential_privacy:
        #     self.dp_engine.detach_engine()

        # self.total_grads = [g.cpu() for g in total_grads]
        # def getAvgGrad(grads):
        #     avgedgrad = grads[0]
        #     w = args.epochs * len(train_data)
        #     for k in len(avgedgrad):
        #         for i, grad in enumerate(grads):
        #             if i==0:
        #                 avgedgrad[k]=grad[k] / w
        #             else:
        #                 avgedgrad[k]+=grad[k] / w
        #     return avgedgrad
        # self.avgedgrad = getAvgGrad(grads)

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
