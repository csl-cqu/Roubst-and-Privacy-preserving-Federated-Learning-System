from mpi4py import MPI

from modules.backdoor_defense.utils.mnist import MaliciousTrainer
from modules.differential_privacy import defense
from spfl_api.utils.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from .FedAVGAggregator import FedAVGAggregator
from .FedAVGTrainer import FedAVGTrainer
from .FedAvgClientManager import FedAVGClientManager
from .FedAvgServerManager import FedAVGServerManager


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedAvg_distributed(process_id, worker_number, device, comm, model, train_data_num, train_data_global,
                             test_data_global,
                             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args,
                             model_trainer=None, preprocessed_sampling_lists=None):
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, model, train_data_num, train_data_global,
                    test_data_global, train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                    model_trainer, preprocessed_sampling_lists)
    else:
        if args.differential_privacy:
            dp_module = defense.Differential_Privacy_Module(model=model, distributed=True)
        else:
            dp_module = None
        init_client(args, device, comm, process_id, worker_number, model, train_data_num, train_data_local_num_dict,
                    train_data_local_dict, test_data_local_dict, model_trainer, dp_module=dp_module)


def init_server(args, device, comm, rank, size, model, train_data_num, train_data_global, test_data_global,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, model_trainer,
                preprocessed_sampling_lists=None):
    if model_trainer is None:
        # default model trainer is for classification problem
        model_trainer = MyModelTrainerCLS(model)
    model_trainer.set_id(-1)

    # aggregator
    worker_num = size - 1
    aggregator = FedAVGAggregator(train_data_global, test_data_global, train_data_num,
                                  train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                                  worker_num, device, args, model_trainer)

    # start the distributed training
    backend = args.backend
    if preprocessed_sampling_lists is None:
        server_manager = FedAVGServerManager(args, aggregator, comm, rank, size, backend)
    else:
        server_manager = FedAVGServerManager(args, aggregator, comm, rank, size, backend,
                                             is_preprocessed=True,
                                             preprocessed_client_lists=preprocessed_sampling_lists)
    server_manager.send_init_msg()
    server_manager.run()


def init_client(args, device, comm, process_id, size, model, train_data_num, train_data_local_num_dict,
                train_data_local_dict, test_data_local_dict, model_trainer=None, dp_module=None):
    client_index = process_id - 1
    if model_trainer is None:
        model_trainer = MyModelTrainerCLS(model, dp_module=dp_module)
    model_trainer.set_id(client_index)
    backend = args.backend
    if args.backdoor_test:
        if args.dataset == 'mnist':
            malicious_trainer = MaliciousTrainer
        else:
            raise NotImplemented()
        trainer = malicious_trainer(client_index, train_data_local_dict, train_data_local_num_dict,
                                    test_data_local_dict, train_data_num, device, args, model_trainer)
    else:
        trainer = FedAVGTrainer(client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                                train_data_num, device, args, model_trainer)
    client_manager = FedAVGClientManager(args, trainer, comm, process_id, size, backend)
    client_manager.run()
