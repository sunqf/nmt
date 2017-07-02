import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import time


class TiedLanguageModel(torch.nn.Module):
    def __init__(self, num_token, dim, rnn_type, num_layers, dropout=0.5):
        super(TiedLanguageModel, self).__init__()

        self.dropout = torch.nn.Dropout(dropout)

        self.encoder = torch.nn.Embedding(num_token, dim, padding_idx=0)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(torch.nn, rnn_type)(dim, dim, num_layers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for '--model' was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = torch.nn.RNN(dim, dim, num_layers, nonlinearity=nonlinearity, dropout=dropout)

        self.decoder = nn.Linear(dim, num_token)

        self.decoder.weight = self.encoder.weight

        self.rnn_type = rnn_type
        self.dim = dim
        self.num_layers = num_layers

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, input, hidden):
        emb = self.dropout(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.dropout(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), -1), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers, batch_size, self.dim).zero_()),
                    Variable(weight.new(self.num_layers, batch_size, self.dim).zero_()))
        else:
            Variable(weight.new(self.num_layers, batch_size, self.dim).zero_())


class Dict(object):
    def __init__(self, words):
        self.words = words

        self.words.insert(0, 'padding')
        self.words.insert(1, 'unk')
        self.words.insert(2, 'eos')
        self.padding_idx = 0
        self.unk_idx = 1
        self.eos = 2
        self.word2ids = {word: id for id, word in enumerate(self.words)}

    def getPaddingIdx(self):
        return self.padding_idx

    def getUnkIdx(self):
        return self.unk_idx

    def vocabSize(self):
        return len(self.words)

    def getId(self, word):
        return self.word2ids.get(word, self.unk_idx)

    def ids(self, words):
        return [self.word2ids.get(word, self.unk_idx) for word in words]

    @staticmethod
    def build(corpus_paths, vocab_size):
        import collections
        word_counts = collections.defaultdict(int)

        for path in corpus_paths:
            assert os.path.exists(path)
            with open(path, 'r') as file:
                for line in file:
                    for word in line.split():
                        word_counts[word] += 1

        if len(word_counts) > vocab_size:
            word_counts = sorted(word_counts, key=lambda item: item[1], reverse=True)[:vocab_size]

        return Dict(list(word_counts.keys()))


class Corpus(object):
    def __init__(self, paths, vocab_size=10000):
        self.dict = Dict.build(paths, vocab_size)
        self.vocab_size = self.dict.vocabSize()

    def data(self, path):
        assert os.path.exists(path)
        with open(path, 'r') as file:
            num_tokens = 0
            for line in file:
                num_tokens += len(line.split()) + 1

        ids = torch.LongTensor(num_tokens)
        with open(path, 'r') as file:
            token = 0
            for line in file:
                for word in line.split() + ['eos']:
                    ids[token] = self.dict.getId(word)
                    token += 1

        return ids


class HyperParam(object):
    def __init__(self):
        self.vocab_size = 10000
        self.batch_size = 32
        self.dim = 1024
        self.num_layers = 1
        self.rnn_type = 'LSTM'
        self.dropout = 0.5
        self.cuda = False
        self.bptt = 35
        self.clip = 0.25
        self.lr = 0.1
        self.opt_method = 'adam'
        self.model_prefix = 'model/lm'
        self.logdir = 'log'
        self.epochs = 10


hyperParam = HyperParam()

corpus = Corpus(['penn/train.txt', 'penn/valid.txt', 'penn/test.txt'], hyperParam.vocab_size)

train_data = corpus.data('penn/train.txt')
valid_data = corpus.data('penn/valid.txt')
test_data = corpus.data('penn/test.txt')


def batchify(data, batch_size):
    num_batch = len(data) // batch_size

    data = data.narrow(0, 0, num_batch * batch_size)
    data = data.view(batch_size, -1).t().contiguous()

    return data


log_interval = 100
train_data = batchify(train_data, hyperParam.batch_size)
valid_data = batchify(valid_data, hyperParam.batch_size)
test_data = batchify(test_data, hyperParam.batch_size)

model = TiedLanguageModel(num_token=corpus.vocab_size,
                          dim=hyperParam.dim,
                          num_layers=hyperParam.num_layers,
                          rnn_type=hyperParam.rnn_type,
                          dropout=hyperParam.dropout)

if hyperParam.cuda:
    train_data.cuda()
    valid_data.cuda()
    test_data.cuda()
    model.cuda()

num_token = corpus.vocab_size
criterion = nn.CrossEntropyLoss()


def getBatch(data, i, evaluation=False):
    seqLen = min(hyperParam.bptt, len(data) - 1 - i)
    batch = Variable(data[i:i + seqLen], volatile=evaluation)
    targets = Variable(data[i + 1:i + 1 + seqLen].view(-1))
    return batch, targets


def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def train(epoch, optimizer):
    model.train()
    total_loss = 0

    import time
    start_time = time.time()

    num_token = corpus.vocab_size
    hidden = model.init_hidden(hyperParam.batch_size)

    train_loss_list = []

    for batch, i in enumerate(range(0, train_data.size(0) - 1, hyperParam.bptt)):
        data, targets = getBatch(train_data, i)

        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, num_token), targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), hyperParam.clip)

        optimizer.step()

        total_loss += loss.data

        train_loss_list.append(loss.data[0])

        import math
        if batch % log_interval == 0 and batch > 0:
            curr_loss = total_loss[0] / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // hyperParam.bptt,
                              elapsed * 1000 / log_interval, curr_loss, math.exp(curr_loss)))
            total_loss = 0
            start_time = time.time()

    return train_loss_list


def get_optimizer(model):
    if hyperParam.opt_method == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperParam.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=hyperParam.lr)

    return optimizer


def evaluation(model, data):
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(hyperParam.batch_size)
    for i in range(0, data.size(0) - 1, hyperParam.bptt):
        data, targets = getBatch(data, i)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(output.view(-1, num_token), targets)
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data)


optimizer = get_optimizer(model)
train_loss_list = []
val_loss_list = []
best_val_loss = None
for epoch in range(1, hyperParam.epochs):
    epoch_start_time = time.time()
    train_loss = train(epoch, optimizer)

    train_loss_list += train_loss
    val_loss = evaluation(model, valid_data)
    val_loss_list.append(val_loss)

    if best_val_loss is None or val_loss < best_val_loss:
        with open(hyperParam.model_prefix, 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss
    else:
        for group in optimizer.param_groups():
            group['lr'] /= 4.0

    import numpy as np

    with open(hyperParam.logdir + '/loss.txt', 'wb') as f:
        np.savetxt(f, np.array(train_loss_list))
    with open(hyperParam.logdir + '/val_loss.txt', 'wb') as f:
        np.savetxt(f, np.array(val_loss_list))

test_loss = evaluation(model, test_data)
print('test loss {:5.2f}'.format(test_loss))
