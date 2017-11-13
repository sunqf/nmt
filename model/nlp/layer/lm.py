
from torch.autograd import Variable

import torch
import torch.nn as nn

class LanguageModel(nn.Module):
    def __init__(self, dim, num_vocab, shared_weight, bidirectional=True, dropout=0.5):
        super(LanguageModel, self).__init__()
        self.dim = dim
        self.bidirectional = bidirectional
        self.hidden_dropout = nn.Dropout(dropout)
        self.forward_linear = nn.Linear(dim, dim)

        if self.bidirectional:
            self.backward_linear = nn.Linear(dim, dim)

        self.output_embedding = nn.Linear(dim, num_vocab)
        self.output_embedding.weight = shared_weight

        self.cross_entropy = nn.CrossEntropyLoss(size_average=False)

    def criterion(self, sentences, hidden_states):

        sentences, batch_sizes = sentences
        hidden_states, batch_sizes = hidden_states
        hidden_states = self.hidden_dropout(hidden_states)

        loss = Variable(torch.FloatTensor([0]))
        if self.forward_linear.weight.is_cuda:
            loss = loss.cuda()

        if len(batch_sizes) >= 2:
            total = sum(batch_sizes)

            count = total - batch_sizes[0]
            # forward language model
            context_start = 0
            next_start = batch_sizes[0]
            for i in range(1, len(batch_sizes)):
                context = hidden_states[context_start:context_start + batch_sizes[i], 0:self.dim]
                next = sentences[next_start:next_start + batch_sizes[i]]
                output = self.output_embedding(self.forward_linear(context))
                loss += self.cross_entropy(output, next)
                context_start += batch_sizes[i - 1]
                next_start += batch_sizes[i]

            if self.bidirectional:
                # backward language model
                context_start = total
                next_start = total - batch_sizes[-1]
                for i in range(len(batch_sizes) - 2, -1, -1):
                    context_start -= batch_sizes[i + 1]
                    next_start -= batch_sizes[i]
                    context = hidden_states[context_start:context_start + batch_sizes[i + 1], self.dim:]
                    next = sentences[next_start:next_start + batch_sizes[i + 1]]
                    output = self.output_embedding(self.backward_linear(context))
                    loss += self.cross_entropy(output, next)

                count += total - batch_sizes[0]

            return loss, count
        else:
            return loss, 1
