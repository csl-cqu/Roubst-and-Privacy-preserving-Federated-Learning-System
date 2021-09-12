from abc import ABC, abstractmethod


class ModelTrainer(ABC):
    """Abstract base class for federated learning trainer.
       1. The goal of this abstract class is to be compatible to
       any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
       2. This class can be used in both server and client side
       3. This class is an operator which does not cache any states inside.
    """

    def __init__(self, model, dp_module=None):
        self.model = model
        self.id = 0
        self.dp_module = dp_module

    def set_id(self, trainer_id):
        self.id = trainer_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    @abstractmethod
    def get_model_optimizerparams_grads(self):
        pass

    @abstractmethod
    def set_model_by_names_params(self, names, params):
        pass

    @abstractmethod
    def get_model_namedparams(self):
        pass

    @abstractmethod
    def train(self, train_data, device, args, optimizer):
        pass

    @abstractmethod
    def test(self, test_data, device, args):
        pass

    @abstractmethod
    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        pass
