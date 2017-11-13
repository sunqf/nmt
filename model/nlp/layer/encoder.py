
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from .embedding import Embedding
from .qrnn import QRNN

class Encoder(nn.Module):
    def __init__(self, vocab_size, gazetteers, embedding_dim, hidden_mode, hidden_dim, num_hidden_layer=1,
                 window_sizes=None, dropout=0.5):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.word_embeds = nn.Embedding(vocab_size, self.embedding_dim)
        self.input_embed = Embedding(self.word_embeds, self.embedding_dim, gazetteers)

        self.hidden_dim = hidden_dim
        self.num_hidden_layer = num_hidden_layer
        self.window_sizes = window_sizes
        self.num_direction = 2

        self.hidden_mode = hidden_mode

        if self.hidden_mode == 'QRNN':

            self.hidden_module = QRNN(self.input_embed.output_dim, self.hidden_dim, self.num_hidden_layer,
                                      window_sizes=self.window_sizes, dropout=dropout)
        else:
            self.hidden_module = nn.LSTM(self.input_embed.output_dim, self.hidden_dim, num_layers=self.num_hidden_layer,
                                         bidirectional=True, dropout=dropout)

    def forward(self, input, gazetteers):
        '''
        :param sentence: PackedSequence
        :return: PackedSequence
        '''
        _, batch_sizes = input

        embeds = self.input_embed(input, gazetteers)
        lstm_output, _ = self.hidden_module(embeds)
        return lstm_output

    def output_dim(self):
        return self.hidden_dim * self.num_direction