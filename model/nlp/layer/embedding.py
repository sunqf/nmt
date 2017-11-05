

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import itertools


import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, word_embedding, embedding_dim, gazetteers=None):
        super(Embedding, self).__init__()

        self.word_embedding = word_embedding
        self.embedding_dim = embedding_dim

        self.output_dim = self.embedding_dim

        '''
        if gazetteers is not None:
            self.gazetteers = gazetteers
            self.gazetteers_embeddings = nn.ModuleList(
                [nn.Linear(gazetteer.length(), embedding_dim) for gazetteer in gazetteers])
            self.gazetteers_len = [gazetteer.length() for gazetteer in gazetteers]

            self.gazetteers_index = [0] + list(itertools.accumulate(self.gazetteers_len))[0:-1]
            self.output_dim += self.gazetteer.length()
        '''
        self.output_dim = self.embedding_dim + sum([g.length() for g in gazetteers])

    def forward(self, sentence, gazetteers):
        '''
        :param input: PackedSequence
        :return:
        '''
        sentence, batch_sizes = sentence
        gazetteers, _ = gazetteers

        word_emb = self.word_embedding(sentence)

        '''
        if gazetteers is not None and len(self.gazetteers) > 0:
            gazetteers, batch_sizes = gazetteers

            outputs = [embedding(gazetteers[:, start:start + length])
                       for embedding, (start, length) in zip(self.gazetteers_embeddings,
                                                             zip(self.gazetteers_index, self.gazetteers_len))]
        '''
        output = torch.cat([word_emb, gazetteers], -1)

        return PackedSequence(output, batch_sizes)