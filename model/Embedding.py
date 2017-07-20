
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence

import numpy as np


class PositionEmbedding(nn.Module):
    '''
       http://export.arxiv.org/pdf/1706.03762
       sinusoid position embedding
    '''
    def __init__(self, dim, max_length=5000):
        super(PositionEmbedding, self).__init__()
        position_enc = np.array([
            [pos / np.power(10000, 2*i/dim) for i in range(dim)]
            for pos in range(max_length)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        self.weight = torch.from_numpy(position_enc).type(torch.FloatTensor).cuda()

    def forward(self, input):
        return Variable(self.weight.index(input.data))





class Embedding(nn.Module):

    def __init__(self, word_embedding, emb_dim, feature_dict=None, pos_embedding=None):
        super(Embedding, self).__init__()

        self.word_embedding = word_embedding
        if feature_dict is not None:
            self.feature_dict = feature_dict
            self.feature_embeddings = [nn.Embedding(num_emb, emb_dim, padding_idx=0)
                                       for feature_name, num_emb, emb_dim in feature_dict]

            self.activation = nn.ReLU()
            self.linear = nn.Linear(self.emb_dim + sum([emb_dim for _, _, emb_dim in feature_dict]), emb_dim)

        if pos_embedding is not None:
            self.pos_embedding = pos_embedding


    def forward(self, input):
        '''

        :param input: PackedSequence
        :return:
        '''
        input, batch_sizes = input

        if input.dim() == 3:
            emb = self.word_embedding(input[:, :, 0])
        else:
            emb = self.word_embedding(input)

        if hasattr(self, 'feature_dict'):
            feats = [feature_embedding(input[:, :, i+1]) for i, feature_embedding in enumerate(self.feature_dict)]

            emb = self.activation(self.linear(torch.cat(emb + feats, -1)))

        if hasattr(self, 'pos_embedding'):
            input_pos = torch.cat([torch.LongTensor([pos] * batch_size) for pos, batch_size in enumerate(batch_sizes)])

            pos_emb = self.pos_embedding(Variable(input_pos).cuda())

            emb = emb + pos_emb

        return PackedSequence(emb, batch_sizes)

