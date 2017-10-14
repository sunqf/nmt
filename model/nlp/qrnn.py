import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

import itertools


class Gates(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(Gates, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(input_dim, output_dim * 3, self.kernel_size, padding=(self.kernel_size - 1) // 2)

    def forward(self, input):
        '''
        :param input: batch * input_dim * length
        :return: f, o, z   batch * length * output_dim
        '''
        # batch * length * channel -> batch * channel * length
        input = input.transpose(1, 2).contiguous()
        gates = self.conv(input)
        # batch * channel * length -> batch * length * channel
        gates = gates.transpose(1, 2).contiguous()

        f, o, z = gates.chunk(3, -1)

        f = f.sigmoid()
        o = o.sigmoid()
        z = z.tanh()

        return f, o, z


class QRNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bidirectional=True, dropout=0.):
        super(QRNNLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size

        self.gates = Gates(self.input_dim, self.output_dim, self.kernel_size)

        self.bidirectional = bidirectional
        if self.bidirectional:
            self.reverse_gates = Gates(self.input_dim, self.output_dim, self.kernel_size)

        self.dropout = dropout

    def pool(self, input, f, o, z, init_cell):
        max_batch_size = input.size(0)
        max_length = input.size(1)
        if init_cell is None:
            init_cell = torch.autograd.Variable(
                input.data.new(max_batch_size, self.output_dim).zero_())

        cells = [init_cell]

        for t in range(max_length):
            cell = f[:, t, :] * cells[-1] + (1 - f[:, t, :]) * z[:, t, :]
            cells.append(cell)

        cells = torch.cat([c.unsqueeze(1) for c in cells[1:]], 1)
        output = cells * o
        return output, cells

    def reverse_pool(self, input, f, o, z, init_cell):
        max_batch_size = input.size(0)
        max_length = input.size(1)
        if init_cell is None:
            init_cell = torch.autograd.Variable(
                input.data.new(max_batch_size, self.output_dim).zero_())

        cells = [init_cell]

        for t in range(max_length - 1, -1, -1):
            cell = f[:, t, :] * cells[-1] + (1 - f[:, t, :]) * z[:, t, :]
            cells.append(cell)
        cells = torch.cat([c.unsqueeze(1) for c in reversed(cells[1:])], 1)
        output = cells * o

        return output, cells

    def forward(self, input, lengths, init_cell=None):

        f, o, z = self.gates(self.gate_dropout(input, self.dropout))

        hidden, cell = self.pool(input, f, o, z, init_cell)
        last = (hidden, cell)
        # reverse
        if self.bidirectional:
            f, o, z = self.reverse_gates(self.gate_dropout(input, self.dropout))
            rhidden, rcell = self.reverse_pool(input, f, o, z, init_cell)

            output = torch.cat([hidden, rhidden], -1)
            last = (torch.cat([hidden[:, -1, :], rhidden[:, -1, :]], -1),
                    torch.cat([cell[:, -1, :], rcell[:, -1, :]], -1))

        return output, last

    def gate_dropout(self, input, dropout):

        if self.training and (dropout > 0.):
            mask = Variable(input.data.new(input.size(0), 1, input.size(2)).bernoulli_(1 - dropout).div_(1 - dropout))
            return input * mask.expand_as(input)
        else:
            return input


class QRNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, kernel_sizes=3, bidirectional=True, dropout=0.):
        super(QRNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        if isinstance(kernel_sizes, int) and kernel_sizes % 2 == 1:
            self.kernel_sizes = [kernel_sizes] * num_layers
        elif isinstance(kernel_sizes, list):
            if len(itertools.filterfalse(lambda x: x % 2 == 1, kernel_sizes)) > 0:
                raise Exception("Elements of kernel_sizes should be odd.")

            if len(kernel_sizes) == num_layers and kernel_sizes % 2 == 1:
                self.kernel_sizes = kernel_sizes
            else:
                raise Exception("length of kernel_sizes should be equal to num_layers")
        elif kernel_sizes is None:
            self.kernel_sizes = [3] * num_layers

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        input_dim = self.input_dim
        layers = []
        for l in range(self.num_layers):
            layer = QRNNLayer(input_dim, self.output_dim, self.kernel_sizes[l],
                              bidirectional=self.bidirectional,
                              dropout=dropout)
            layers.append(layer)
            input_dim = self.output_dim * self.num_directions

        self.layers = nn.ModuleList(layers)


    def forward(self, data, init_cell=None):
        is_packed = isinstance(data, PackedSequence)
        if is_packed:
            data, lengths = pad_packed_sequence(data, batch_first=True)
        else:
            raise RuntimeError("Data must be the type of PackedSequence.")

        batch_size = data.size(0)

        lasts = []
        output = data
        for layer in self.layers:
            output, last = layer(output, lengths)
            lasts.append(last)

        hiddens, cells = zip(*lasts)
        hidden = torch.cat(hiddens, 0).view(self.num_directions * self.num_layers, batch_size, self.output_dim)
        cell = torch.cat(cells, 0).view(self.num_directions * self.num_layers, batch_size, self.output_dim)

        return pack_padded_sequence(output, lengths, batch_first=True), (hidden, cell)




