from abc import abstractmethod

from mpi4py import MPI

from ..communication.message import Message
from ..communication.mpi.com_manager import MpiCommunicationManager
from ..communication.observer import Observer


class ClientManager(Observer):

    def __init__(self, args, comm=None, rank=0, size=0, backend="MPI"):
        self.args = args
        self.size = size
        self.rank = rank
        self.backend = backend
        self.com_manager = MpiCommunicationManager(comm, rank, size, node_type="client")
        self.com_manager.add_observer(self)
        self.message_handler_dict = dict()

    def run(self):
        self.register_message_receive_handlers()
        self.com_manager.handle_receive_message()

    def get_sender_id(self):
        return self.rank

    def receive_message(self, msg_type, msg_params) -> None:
        handler_callback_func = self.message_handler_dict[msg_type]
        handler_callback_func(msg_params)

    def send_message(self, message):
        msg = Message()
        msg.add(Message.MSG_ARG_KEY_TYPE, message.get_type())
        msg.add(Message.MSG_ARG_KEY_SENDER, message.get_sender_id())
        msg.add(Message.MSG_ARG_KEY_RECEIVER, message.get_receiver_id())
        for key, value in message.get_params().items():
            msg.add(key, value)
        self.com_manager.send_message(msg)

    @abstractmethod
    def register_message_receive_handlers(self) -> None:
        pass

    def register_message_receive_handler(self, msg_type, handler_callback_func):
        self.message_handler_dict[msg_type] = handler_callback_func

    def finish(self):
        MPI.COMM_WORLD.Abort()
