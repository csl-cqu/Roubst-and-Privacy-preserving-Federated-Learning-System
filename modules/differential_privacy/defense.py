from opacus.dp_model_inspector import DPModelInspector
from modules.differential_privacy.dp_engine import DPML_Engine
from opacus.utils import module_modification


# check if nn model is supported by opacus
def model_check(model):
    if not model:
        return False
    inspector = DPModelInspector()
    if inspector.validate(model):
        return True
    else:
        # add this for trying convert model
        # Model_Utils.model_modifier(model)
        raise Warning("Trying to modify model for opacus")


def model_modifier(model):
    # convert
    model = module_modification.convert_batchnorm_modules(model)
    # check again
    inspector = DPModelInspector()
    flag = inspector.validate(model)
    return flag


class Differential_Privacy_Module:
    def __init__(self, model, distributed=False):
        self.distributed = distributed
        self.attached = False
        # model_modifier(model)
        # check
        if not model_check(model):
            raise Warning("model is incompatible")
        self.ENGINE_DIR = {}  # dir of differential privacy engine

    # attach new engine
    def init_new_engine(self, model, optimizer, device, client,
                        epochs, sample_size, batch_size, sigma, delta, grad_norm):
        if self.ENGINE_DIR.get(client, None) is not None:  # do not init twice
            return

        engine = DPML_Engine(model, client, epochs, sample_size, batch_size, sigma, delta, grad_norm)
        engine.attach_optimizer(optimizer, device)
        # save handle
        self.ENGINE_DIR[client] = engine

        if self.distributed is True:
            self.attached = True

    # return boolean
    def not_attached(self):
        if self.distributed and self.attached is False:
            return True
        else:
            return False

    # update single engine
    def update_engine_spent(self, client, log, round_idx=None):
        self.ENGINE_DIR[client].update_spent(log=log, round_idx=round_idx)

    # update all engines in manager module
    def update_all_spent(self, log, round_idx=None):
        for k in self.ENGINE_DIR.keys():
            self.ENGINE_DIR[k].update_spent(log=log, round_idx=round_idx)

    # debug
    def update_all_spent_output(self, log, round_idx=None):
        for k in self.ENGINE_DIR.keys():
            self.ENGINE_DIR[k].update_spent(log=log, round_idx=round_idx)
            self.ENGINE_DIR[k].output_spent()

    # debug
    def output_spent_list(self):
        for k in self.ENGINE_DIR.keys():
            self.ENGINE_DIR[k].output_spent()
