import torch

from torch.optim import Optimizer


class SGD(Optimizer):
    r"""
    Modified version of SGD optimizer which supports gradients.
    """

    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, grads, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for i, p in enumerate(group['params']):
                params_with_grad.append(p)
                d_p_list.append(grads[i])

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

                if weight_decay != 0:
                    grads[i] = grads[i].add(p, alpha=weight_decay)

                if momentum != 0:
                    buf = momentum_buffer_list[i]

                    if buf is None:
                        buf = torch.clone(grads[i]).detach()
                        momentum_buffer_list[i] = buf
                    else:
                        buf.mul_(momentum).add_(grads[i], alpha=1 - dampening)

                    if nesterov:
                        grads[i] = grads[i].add(buf, alpha=momentum)
                    else:
                        grads[i] = buf

                p.data.add_(grads[i], alpha=-lr)

                state = self.state[p]
                state['momentum_buffer'] = state['momentum_buffer'] if 'momentum_buffer' in state else None

        return loss
