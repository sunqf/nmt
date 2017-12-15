
import torch.nn as nn


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()

        self.use_cuda = False

    def cuda(self, device):
        self.use_cuda = True
        for module in self.children():
            if isinstance(module, Module):
                module.use_cuda = True
        return super(Module, self).cuda(device)


    def cpu(self):
        self.use_cuda = False
        return super(Module, self).cpu()



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