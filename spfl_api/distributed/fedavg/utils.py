import logging
import os
import sys

import numpy as np
import torch


def transform_list_to_tensor(model_params_list):
    '''
    server/client do this on received model params messages, if is_mobile==1
    :param model_params_list: model.state_dict
    '''
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(np.asarray(model_params_list[k])).float()
    return model_params_list


def transform_params_list_to_tensor(model_params_list):
    '''
    server/client do this on received model params messages, if is_mobile==1
    :param model_params_list: model.parameters
    '''
    for k in range(len(model_params_list)):
        model_params_list[k] = torch.from_numpy(np.asarray(model_params_list[k])).float()
    return model_params_list


def transform_tensor_to_list(model_params):
    '''
    server/client do this before sending model params messages, if is_mobile==1.
    :param model_params: model.state_dict
    '''
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params


def transform_params_tensor_to_list(model_params):
    '''
    server/client do this before sending model params messages, if is_mobile==1.
    :param model_params: model.state_dict
    '''
    for k in range(len(model_params)):
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params


def post_complete_message_to_sweep_process(args):
    # TODO can be skipped?
    logging.warning("post_complete_message_to_sweep_process is skipped")

    return

    base_path = './tmp'
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    pipe_path = base_path + "/fedml"

    def windows_pipe():
        raise NotImplementedError

    def linux_pipe():
        if not os.path.exists(pipe_path):
            os.mkfifo(pipe_path)
        pipe_fd = os.open(pipe_path, os.O_WRONLY)
        with os.fdopen(pipe_fd, 'w') as pipe:
            pipe.write("training is finished! \n%s\n" % (str(args)))

    if sys.platform == 'win32':
        try:
            windows_pipe()
        except NotImplementedError:
            logging.warning("pipe code is not implemented in windows.")
    else:
        linux_pipe()
