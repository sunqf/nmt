
import torch.nn as nn


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()

        self.use_cuda = False

    def cuda(self, device):
        super(Task, self).cuda(device)
        self.use_cuda = True

    def cpu(self):
        super(Task, self).cpu()
        self.use_cuda = False


class Task(Module):

    def __init__(self):
        super(Task, self).__init__()

    def loss(self, batch_data):
        pass

    def evaluation(self, data):
        pass

    def sample(self, batch_data):
        pass


class Loader:

    def batch_train(self, vocab, gazetters, batch_size):
        pass

    def batch_test(self, vocab, gazetters, batch_size):
        pass

class TaskConfig:
    def loader(self):
        pass

    def create_task(self, shared_vocab, shared_encoder):
        pass