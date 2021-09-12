import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
DATA_CIFAR10 = os.path.join(DATA, "cifar10")
DATA_CIFAR100 = os.path.join(DATA, "cifar100")
DATA_MNIST = os.path.join(DATA, "mnist")
DATA_MNIST_ORIGINAL = os.path.join(DATA, "mnist_original")
MODULES = os.path.join(ROOT, "modules")
MODULE_DATA_RECOV = os.path.join(MODULES, "data_inference_defense")
MAIN_FEDAVG = os.path.join(ROOT, "spfl_experiments/distributed/fedavg/main_fedavg.py")