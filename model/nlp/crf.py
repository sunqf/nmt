
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import itertools


import torch
import torch.nn as nn

def log_sum_exp(vecs, axis):
    max_val, _ = vecs.max(axis)
    vecs = vecs - max_val.unsqueeze(axis)
    out_val = torch.log(torch.exp(vecs).sum(axis))
    # print(max_val, out_val)
    return max_val + out_val


class Embedding(nn.Module):
    def __init__(self, word_embedding, embedding_dim, gazetteers=None):
        super(Embedding, self).__init__()

        self.word_embedding = word_embedding
        self.embedding_dim = embedding_dim

        self.output_dim = self.embedding_dim

        if gazetteers is not None:
            self.gazetteers = gazetteers
            self.gazetteers_embeddings = nn.ModuleList(
                [nn.Linear(gazetteer.length(), embedding_dim) for gazetteer in gazetteers])
            self.gazetteers_len = [gazetteer.length() for gazetteer in gazetteers]

            self.gazetteers_index = [0] + list(itertools.accumulate(self.gazetteers_len))[0:-1]
            self.output_dim += self.embedding_dim

    def output_dim(self):
        return self.output_dim

    def forward(self, sentence, gazetteers):
        '''
        :param input: PackedSequence
        :return:
        '''
        sentence, batch_sizes = sentence

        word_emb = self.word_embedding(sentence)

        if gazetteers is not None and len(self.gazetteers) > 0:
            gazetteers, batch_sizes = gazetteers

            outputs = [embedding(gazetteers[:, start:start + length]).unsqueeze(-1)
                       for embedding, (start, length) in zip(self.gazetteers_embeddings,
                                                             zip(self.gazetteers_index, self.gazetteers_len))]

        output = word_emb + torch.cat([word_emb, *outputs], -1).sum(-1)

        return PackedSequence(output, batch_sizes)


class CRFLayer(nn.Module):
    def __init__(self, feature_dim, num_labels, dropout=0.5):
        super(CRFLayer, self).__init__()
        self.hidden_dim = feature_dim
        self.num_labels = num_labels
        self.feature_dropout = nn.Dropout(dropout)
        self.feature2labels = nn.Linear(feature_dim, num_labels)
        self.start_transition = nn.Parameter(torch.randn(self.num_labels))
        # tags[i + 1] -> tags[i]
        self.transitions = nn.Parameter(torch.randn(self.num_labels, self.num_labels))
        self.end_transition = nn.Parameter(torch.randn(self.num_labels))

    def _forward_alg(self, emissions):
        '''
        :param emissions: PackedSequence
        :return:
        '''
        emissions, batch_sizes = emissions
        scores = emissions[0:batch_sizes[0]] + self.start_transition
        emission_offset = batch_sizes[0]
        # Get the log sum exp score
        for i in range(1, len(batch_sizes)):
            scores[:batch_sizes[i]] = emissions[emission_offset:emission_offset + batch_sizes[i]] + \
                                      log_sum_exp(scores[:batch_sizes[i]].view(batch_sizes[i], 1,
                                                                               self.num_labels) + self.transitions,
                                                  axis=-1)
            emission_offset += batch_sizes[i]
        scores = scores + self.end_transition
        return log_sum_exp(scores, axis=-1)

    def _emission_select(self, emissions, batch_size, tags):
        return emissions.gather(-1, tags[:batch_size].unsqueeze(-1)).squeeze(-1)

    def _transition_select(self, batch_size, prev_tags, curr_tags):
        return self.transitions.index_select(0, curr_tags).gather(1, prev_tags.unsqueeze(-1)).squeeze(-1)

    def _score_sentence(self, emissions, tags):
        '''

        :param emissions: packedsequence
        :param tags: packedsequence
        :param batch_sizes:
        :return:
        '''
        emissions, batch_sizes = emissions
        tags, _ = tags
        last_tags = tags[:batch_sizes[0]]
        score = self.start_transition.gather(0, tags[0:batch_sizes[0]]) + \
                self._emission_select(emissions[0:batch_sizes[0]], batch_sizes[0], last_tags)

        emissions_offset = batch_sizes[0]
        for i in range(1, len(batch_sizes)):
            curr_tags = tags[emissions_offset:emissions_offset + batch_sizes[i]]
            score[:batch_sizes[i]] = score[:batch_sizes[i]] + \
                                     self._transition_select(batch_sizes[i], last_tags[:batch_sizes[i]], curr_tags) + \
                                     self._emission_select(
                                         emissions[emissions_offset:emissions_offset + batch_sizes[i]], batch_sizes[i],
                                         curr_tags)
            last_tags = last_tags.clone()
            last_tags[:batch_sizes[i]] = curr_tags
            emissions_offset += batch_sizes[i]
        score = score + self.end_transition.gather(0, last_tags)
        return score

    def _viterbi_decode(self, emissions):
        '''

        :param emissions: [len, label_size]
        :return:
        '''
        emissions = emissions.data.cpu()
        scores = torch.zeros(emissions.size(-1))
        back_pointers = torch.zeros(emissions.size()).int()
        transitions = self.transitions.data.cpu()
        scores = scores + self.start_transition.data.cpu() + emissions[0]
        # Generate most likely scores and paths for each step in sequence
        for i in range(1, emissions.size(0)):
            scores_with_transitions = scores + transitions
            max_scores, back_pointers[i] = torch.max(scores_with_transitions, -1)
            scores = emissions[i] + max_scores
        # Generate the most likely path
        scores = scores + self.end_transition.data.cpu()
        viterbi = [scores.numpy().argmax()]
        back_pointers = back_pointers.numpy()
        for bp in reversed(back_pointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()
        viterbi_score = scores.numpy().max()
        return viterbi_score, viterbi

    def neg_log_likelihood(self, feats, tags):
        '''

        :param feats: PackedSequence
        :param tags: PackedSequence
        :return:
        '''
        feats, batch_sizes = feats
        feats = self.feature_dropout(feats)
        emissions = PackedSequence(self.feature2labels(feats), batch_sizes)
        forward_score = self._forward_alg(emissions)
        gold_score = self._score_sentence(emissions, tags)
        return (forward_score - gold_score).sum()

    def forward(self, feats):
        '''
        unsupported batch process
        :param feats: PackedSequence
        :return: score, tag sequence
        '''
        # Find the best path, given the features.
        feats, batch_sizes = feats
        feats = self.feature_dropout(feats)
        emissions = PackedSequence(self.feature2labels(feats), batch_sizes)
        sentences, lens = pad_packed_sequence(emissions, batch_first=True)
        return [self._viterbi_decode(sentence[:len]) for sentence, len in zip(sentences, lens)]


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

            denom = total - batch_sizes[0]
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

                denom += total - batch_sizes[0]

            return loss / denom
        else:
            return loss


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, num_label, gazetteers, embedding_dim, hidden_mode, hidden_dim, num_hidden_layer=1,
                 window_sizes=None, dropout=0.5):
        super(BiLSTMCRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.word_embeds = nn.Embedding(vocab_size, self.embedding_dim)
        self.input_embed = Embedding(self.word_embeds, self.embedding_dim, gazetteers)

        self.hidden_dim = hidden_dim
        self.num_hidden_layer = num_hidden_layer
        self.window_sizes = window_sizes
        self.num_direction = 2

        self.hidden_mode = hidden_mode

        if self.hidden_mode == 'QRNN':
            from .qrnn import QRNN
            self.hidden_module = QRNN(self.embedding_dim, self.hidden_dim, self.num_hidden_layer,
                                      window_sizes=self.window_sizes, dropout=dropout)
        else:
            self.hidden_module = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_hidden_layer,
                                         bidirectional=True, dropout=dropout)

        # output layer
        self.crf = CRFLayer(embedding_dim * 2, num_label, dropout)

    def _get_features(self, input, gazetteers):
        '''
        :param sentence: PackedSequence
        :return: [seq_len, hidden_dim * 2]
        '''
        _, batch_sizes = input

        embeds = self.input_embed(input, gazetteers)
        lstm_output, _ = self.hidden_module(embeds)
        return lstm_output

    def forward(self, sentences, gazetteers):
        return self.crf(self._get_features(sentences, gazetteers))

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_features(sentence)
        return self.crf.neg_log_likelihood(feats, tags)

    def loss(self, sentence, gazetteers, tags):
        feats = self._get_features(sentence, gazetteers)
        return self.crf.neg_log_likelihood(feats, tags)

    # 精调时锁定word embedding,
    def parameters(self):
        for name, parameter in self.named_parameters():
            if name != 'word_embeds':
                yield parameter
