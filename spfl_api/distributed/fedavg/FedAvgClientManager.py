import logging
import os
import sys

from spfl_api.distributed.fedavg.FedAVGTrainer import FedAVGTrainer

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

from spfl_core.distributed.client.client_manager import ClientManager
from spfl_core.distributed.communication.message import Message
from .message_define import MyMessage
from .utils import transform_list_to_tensor


class FedAVGClientManager(ClientManager):
    def __init__(self, args, trainer: FedAVGTrainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

        self.args = args

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)

    # client starts from here
    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        if self.args.is_mobile == 1:
            global_model_params = transform_list_to_tensor(global_model_params)

        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index))
        self.round_idx = 0
        self.__train()

    def start_training(self):
        self.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)

        self.trainer.update_model(model_params)
        self.trainer.update_dataset(int(client_index))
        self.round_idx += 1
        self.__train()

    def send_model_to_server(self, receive_id, local_sample_num, **kwargs):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        if not self.args.use_gradient:
            assert 'weights' in kwargs.keys()
            weights = kwargs['weights']
            message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        else:
            if self.args.byzantine_aggregate and self.args.backdoor_defense:
                assert 'grads' in kwargs.keys() and 'optimizerparams' in kwargs.keys()
                grads = kwargs['grads']
                optimizerparams = kwargs['optimizerparams']
                message.add_params(MyMessage.MSG_ARG_KEY_MODEL_OPTIMIZERPARAMS, optimizerparams)
                message.add_params(MyMessage.MSG_ARG_KEY_MODEL_GRADS, grads)
            else:
                assert 'grads' in kwargs.keys()
                grads = kwargs['grads']
                message.add_params(MyMessage.MSG_ARG_KEY_MODEL_GRADS, grads)

        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def __train(self):
        if self.args.use_gradient:
            if self.args.byzantine_aggregate and self.args.backdoor_defense:
                param, grads, local_sample_num = self.trainer.train(self.round_idx)
                self.send_model_to_server(0, local_sample_num, optimizerparams=param, grads=grads)
            else:
                grads, local_sample_num = self.trainer.train(self.round_idx)
                self.send_model_to_server(0, local_sample_num, grads=grads)
        else:
            weights, local_sample_num = self.trainer.train(self.round_idx)
            self.send_model_to_server(0, local_sample_num, weights=weights)
