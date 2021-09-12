from opacus import PrivacyEngine
from modules.differential_privacy.eps_log import *


class DPML_Engine:
    def __init__(self, model, client, epochs, sample_size, batch_size, sigma, delta, grad_norm):
        self.IDX = client  # client id
        self.privacy_engine = None  # main engine
        self.epochs = epochs  # train epochs
        self.SAMPLE_SIZE = sample_size  #
        self.BATCH_SIZE = batch_size  # sample rate = batch size / sample size
        self.SIGMA = sigma  # noise
        self.DELTA = delta  # target err param
        self.MAX_GRAD_NORM = grad_norm  # clipper param

        self.privacy_engine = self.init_privacy_engine(model)
        self.optimizer = None
        self.spent_save = 0

        params = [{'epochs': self.epochs, 'sample_size': self.SAMPLE_SIZE, 'BATCH_SIZE': self.BATCH_SIZE,
                   'sigma': self.SIGMA, 'delta': self.DELTA, 'grad_norm': self.MAX_GRAD_NORM}]
        self.log = DPLog(self.IDX, params)
        self.log.create_log_file()

    # attach engine to model, optimizer
    # param device is used for model modify
    def attach_optimizer(self, optimizer, device):
        # convert model for opacus API
        # model = module_modifier(model)
        # model = model.to(device)
        # try:
        self.privacy_engine.attach(optimizer)
        self.optimizer = optimizer
        # except ValueError as e1:
        #     pass
        # except AttributeError as e2:
        #     pass
        # DEBUG
        # print("if optim.dp is not none: ", (optimizer.privacy_engine is not None))

    # init the privacy engine
    def init_privacy_engine(self, model):
        # init param sample rate with virtual batch
        # SAMPLE_RATE = self.BATCH_SIZE / self.len_train_dataset
        # assert self.VIRTUAL_BATCH_SIZE % self.BATCH_SIZE == 0
        # VIRTUAL_BATCH_SIZE should be divisible by BATCH_SIZE
        # self.N_ACCUMULATION_STEPS = int(self.VIRTUAL_BATCH_SIZE / self.BATCH_SIZE)
        # print(f"(sample_size = {self.SAMPLE_SIZE}, batch_size = {self.BATCH_SIZE})")

        clipping = {"clip_per_layer": False, "enable_stat": True}
        if self.SAMPLE_SIZE < self.BATCH_SIZE:
            self.SAMPLE_SIZE = self.BATCH_SIZE

        return PrivacyEngine(
            model,  # torch nn model
            epochs=self.epochs,
            batch_size=self.BATCH_SIZE,  # choose batch size and sample size
            sample_size=self.SAMPLE_SIZE,
            # sample_rate=10 / 60000,       # or choose sample rate init
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            # should be import with args
            noise_multiplier=self.SIGMA,
            # could be cancelled when less than other modules
            max_grad_norm=self.MAX_GRAD_NORM,  # clip grad
            # max_grad_norm=None,
            # supports a CSPRNG provided by the torchcsprng library
            # cryptographically secure RNGs
            secure_rng=True,
            **clipping
        )

    def update_spent(self, log, round_idx=None):
        try:
            epsilon, best_alpha = self.optimizer.privacy_engine.get_privacy_spent(target_delta=self.DELTA)
            self.spent_save = epsilon
            if log:
                self.log.save_epsilon(self.IDX, self.spent_save)

        except AttributeError as e:
            pass

    def output_spent(self):
        if self.spent_save != 0:
            print(f"client_idx = {self.IDX}\t(ε = {self.spent_save:.2f}, δ = {self.DELTA})")

    # DEBUG
    # param out for print console
    def print_privacy_spent_info(self, optimizer, out):
        # print("self.DELTA: ", self.DELTA)
        # return the privacy spent in training
        try:
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(target_delta=self.DELTA)
            if out:
                print(f"(ε = {epsilon:.2f}, δ = {self.DELTA})")
        except AttributeError as e:
            pass

    # detach model, optimizer
    # call it when round end
    # def detach_engine(self):
    #    self.privacy_engine.detach()
