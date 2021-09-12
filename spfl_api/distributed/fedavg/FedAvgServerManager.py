import datetime
import os
import sys

import torch

from modules.backdoor_defense.defense import when_server_receive_model_update
from .message_define import MyMessage
from .utils import transform_tensor_to_list, post_complete_message_to_sweep_process

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

from spfl_core.distributed.communication.message import Message
from spfl_core.distributed.server.server_manager import ServerManager
from .FedAVGAggregator import FedAVGAggregator


class FedAVGServerManager(ServerManager):
    def __init__(self, args, aggregator: FedAVGAggregator, comm=None, rank=0, size=0, backend="MPI",
                 is_preprocessed=False, preprocessed_client_lists=None):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists

    def run(self):
        super().run()

    def save_to_localfile(self):
        # save model to local file
        if not self.args.no_save_to_localfile:
            if self.args.save_path:
                model_path = self.args.save_path if self.args.save_path.endswith('.pth') else self.args.save_path + '.pth'
            else:
                t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = './{}.pth'.format(self.args.wandb_name)
            torch.save(self.aggregator.trainer.model.state_dict(), model_path)

    def send_init_msg(self):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        global_model_params = self.aggregator.get_global_model_params()
        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params, client_indexes[process_id - 1])

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        if not self.args.use_gradient:
            weights = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
            if self.args.backdoor_defense:
                weights = when_server_receive_model_update(weights, self.aggregator.get_global_model_params(),
                                                           self.args.backdoor_defense_shrink,
                                                           self.args.backdoor_defense_noise)
            self.aggregator.add_local_trained_result(sender_id - 1, weights, local_sample_number)
        else:
            if self.args.byzantine_aggregate and self.args.backdoor_defense:
                optimizerparams = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_OPTIMIZERPARAMS)
                grads = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_GRADS)
                self.aggregator.add_local_trained_optimizerparam_grads(sender_id - 1, optimizerparams, grads,
                                                                       local_sample_number)

            else:
                grads = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_GRADS)
                self.aggregator.add_local_trained_grads(sender_id - 1, grads, local_sample_number)

        b_all_received = self.aggregator.check_whether_all_receive()
        if b_all_received:
            global_model_params = self.aggregator.aggregate()
            self.aggregator.test_on_server_for_all_clients(self.round_idx)

            if self.args.backdoor_test:
                t = self.aggregator.train_data_local_dict, self.aggregator.test_global, self.aggregator.val_global
                self.aggregator.train_data_local_dict = self.aggregator.malicious_train_data
                self.aggregator.test_global = self.aggregator.malicious_test_data
                self.aggregator.val_global = self.aggregator.malicious_val_data
                self.aggregator.test_on_server_for_all_clients(self.round_idx, prefix='Backdoor ')
                self.aggregator.train_data_local_dict, self.aggregator.test_global, self.aggregator.val_global = t

            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                post_complete_message_to_sweep_process(self.args)
                self.save_to_localfile()
                self.finish()
                return
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.round_idx]
            else:
                # sampling clients
                client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                                 self.args.client_num_per_round)
            if self.args.is_mobile == 1:
                global_model_params = transform_tensor_to_list(global_model_params)

            for receiver_id in range(1, self.size):
                self.send_message_sync_model_to_client(receiver_id, global_model_params,
                                                       client_indexes[receiver_id - 1])

    def send_message_init_config(self, receive_id, global_model_params, client_index):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index):
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)
