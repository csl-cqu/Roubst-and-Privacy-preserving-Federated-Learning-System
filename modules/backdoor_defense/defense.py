import torch
import logging


def when_server_receive_model_update(weights, origin, shrink, noise):
    for key, val in weights.items():
        delta = val - origin[key]
        logging.info("NORM[" + key + "] = " + str(delta.norm()))
        weights[key] = origin[key] + delta / max(1, delta.norm() / shrink) + noise * torch.randn(delta.shape)
    return weights
