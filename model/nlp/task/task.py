

from ..layer.crf import CRFLayer

import torch.nn as nn
from ..layer.qrnn import QRNN
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence


class Task(nn.Module):
    def __init__(self):
        super(Task, self).__init__()

    def loss(self, batch_data, use_cuda=False):
        pass

    def evaluation(self, data, use_cuda=False):
        pass

    def sample(self, batch_data, use_cuda=False):
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