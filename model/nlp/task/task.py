

from ..layer.crf import CRFLayer

import torch.nn as nn
from ..layer.qrnn import QRNN
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence


class Task(nn.Module):
    def __init__(self):
        super(Task, self).__init__()

    def loss(self, encoder, data):
        pass

    def evaluation(self, encoder, data):
        pass

    def sample(self, encoder, data):
        pass


class DataSet:

    def word_counts(self):
        pass

    def train_batch(self, transformer, batch_size):
        pass

    def test_batch(self, transformer, batch_size):
        pass