
import os
import math
import itertools
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from torch.autograd import Variable

paths = ['/Users/sunqf/startup/quotesbot/nlp-data/chinese_segment/data/train/train.all']
'''
def split_to_single(text):
    for word in text.split():
        for ch in word:
            yield(ch)

def count_word(corpus_paths):

    word_counts = collections.defaultdict(int)

    for path in corpus_paths:
        assert os.path.exists(path)
        with open(path, 'r') as file:
            for line in file:
                for word in split_to_single(line):
                    word_counts[word] += 1
    return word_counts


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
    def build(word_counts, vocab_size):
        if len(word_counts) > vocab_size:
            word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)[:vocab_size]

        return Dict(list([word for word, count in word_counts]))




word_counts = count_word(paths)

with open("word-counts.txt", 'w') as file:
    for k, v in sorted(word_counts.items(), key=lambda items: items[1], reverse=True):
        file.write('%s\t%d\n' % (k, v))


dict = Dict.build(word_counts, vocab_size=5000)

def tag(pos, seq_len):
    if pos == 0:
        return 'B'
    elif pos == seq_len - 1:
        return 'E'
    else:
        return 'I'

def load(corpus_paths):
    for path in corpus_paths:
        assert os.path.exists(path)
        with open(path, 'r') as file:
            for line in file:
                if len(line.strip()) > 0:
                    chars = [ch for word in line.split() for ch in word]
                    tags = [tag(pos, len(word)) for word in line.split() for pos in range(len(word))]
                    assert(len(chars) == len(tags))
                    yield chars, tags


training_data = list(load(paths, dict))
import random
random.shuffle(training_data)
'''

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

def log_sum_exp(vecs, axis):
    max_val, _ = vecs.max(axis)
    vecs = vecs - max_val.unsqueeze(axis)
    out_val = torch.log(torch.exp(vecs).sum(axis))
    # print(max_val, out_val)
    return max_val + out_val

class Embedding(nn.Module):

    def __init__(self, word_embedding, emb_dim, feature_dict=None):
        super(Embedding, self).__init__()

        self.word_embedding = word_embedding
        if feature_dict is not None:
            self.feature_dict = feature_dict
            self.feature_embeddings = [nn.Embedding(num_emb, emb_dim, padding_idx=0)
                                       for feature_name, num_emb, emb_dim in feature_dict]

            self.activation = nn.ReLU()
            self.linear = nn.Linear(self.emb_dim + sum([emb_dim for _, _, emb_dim in feature_dict]), emb_dim)


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

        return PackedSequence(emb, batch_sizes)



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
            scores[:batch_sizes[i]] = emissions[emission_offset:emission_offset+batch_sizes[i]] + \
                                      log_sum_exp(scores[:batch_sizes[i]].view(batch_sizes[i], 1, self.num_labels) + self.transitions, axis=-1)
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
            curr_tags = tags[emissions_offset:emissions_offset+batch_sizes[i]]
            score[:batch_sizes[i]] = score[:batch_sizes[i]] + \
                                     self._transition_select(batch_sizes[i], last_tags[:batch_sizes[i]], curr_tags) + \
                                     self._emission_select(emissions[emissions_offset:emissions_offset+batch_sizes[i]], batch_sizes[i], curr_tags)
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
        #feats = self.feature_dropout(feats)
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
        #feats = self.feature_dropout(feats)
        emissions = PackedSequence(self.feature2labels(feats), batch_sizes)
        sentences, lens = pad_packed_sequence(emissions, batch_first=True)
        return [self._viterbi_decode(sentence[:len]) for sentence, len in zip(sentences, lens)]

class LanguageModel(nn.Module):

    def __init__(self, dim, num_vocab, shared_weight, bidirectional=True, dropout=0.5):
        super(LanguageModel, self).__init__()
        self.dim = dim
        self.bidirectional = bidirectional
        self.hidden_dropout = nn.Dropout(dropout)
        self.forward_linear = nn.Linear(dim, num_vocab)
        self.forward_linear.weight = shared_weight


        if self.bidirectional:
            self.backward_linear = nn.Linear(dim, num_vocab)
            self.backward_linear.weight = shared_weight
        self.cross_entropy = nn.CrossEntropyLoss(size_average=False)


    def criterion(self, sentences, hidden_states):

        sentences, batch_sizes = sentences
        hidden_states, batch_sizes = hidden_states
        #hidden_states = self.hidden_dropout(hidden_states)

        loss = Variable(torch.FloatTensor([0]))
        if self.forward_linear.weight.is_cuda:
            loss = loss.cuda()

        if len(batch_sizes) >= 2:
            total = sum(batch_sizes)

            # forward language model
            context_start = 0
            next_start = batch_sizes[0]
            for i in range(1, len(batch_sizes)):
                context = hidden_states[context_start:context_start + batch_sizes[i], 0:self.dim]
                next = sentences[next_start:next_start+batch_sizes[i]]
                loss += self.cross_entropy(self.forward_linear(context), next)
                context_start += batch_sizes[i-1]
                next_start += batch_sizes[i]


            if self.bidirectional:
                # backward language model
                context_start = total
                next_start = total - batch_sizes[-1]
                for i in range(len(batch_sizes) - 2, 0, -1):
                    context_start -= batch_sizes[i+1]
                    next_start -= batch_sizes[i]
                    context = hidden_states[context_start:context_start + batch_sizes[i+1], self.dim:]
                    next = sentences[next_start:next_start+batch_sizes[i+1]]
                    loss += self.cross_entropy(self.backward_linear(context), next)

            return loss/(total*2-batch_sizes[0]-batch_sizes[-1])
        else:
            return loss


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, num_label, embedding_dim, num_hidden_layer=1, dropout=0.5):
        super(BiLSTMCRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.input_embed = Embedding(self.word_embeds, self.embedding_dim)

        self.num_hidden_layer = num_hidden_layer
        self.num_direction = 2
        self.lstm = nn.LSTM(embedding_dim, embedding_dim,
                            num_layers=num_hidden_layer, bidirectional=True, dropout=dropout)
        self.init_hidden_state = nn.Parameter(torch.Tensor(self.num_hidden_layer * self.num_direction, self.embedding_dim))

        # output layer
        self.crf = CRFLayer(embedding_dim * 2, num_label)
        self.lm = LanguageModel(embedding_dim, vocab_size, self.word_embeds.weight, bidirectional=True)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.normal(self.init_hidden_state, 0, 0.5)

    def _get_features(self, input):
        '''
        :param sentence: PackedSequence
        :return: [seq_len, hidden_dim * 2]
        '''
        _, batch_sizes = input
        init_hidden_size = self.init_hidden_state.unsqueeze(1).expand(self.num_hidden_layer*self.num_direction, batch_sizes[0], self.embedding_dim)
        embeds = self.input_embed(input)
        lstm_output, self.hidden = self.lstm(embeds, (init_hidden_size, init_hidden_size))
        return lstm_output

    def forward(self, sentences):
        return self.crf(self._get_features(sentences))

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_features(sentence)
        return self.crf.neg_log_likelihood(feats, tags)

    def loss(self, sentence, tags):
        feats = self._get_features(sentence)
        return self.crf.neg_log_likelihood(feats, tags), self.lm.criterion(sentence, feats)



torch.set_num_threads(10)

class Config:
    def __init__(self):
        self.max_vocab_size = 5000
        self.batch_size = 16
        self.embedding_size = 128
        self.hidden_size = 128
        self.dropout = 0.3
        self.use_cuda = False

        self.data_root = '/Users/sunqf/startup/quotesbot/nlp-data/chinese_segment/data/'
        #self.data_root = '/home/sunqf/Work/chinese_segment/data'
        self.train_paths = [os.path.join(self.data_root, 'train/train.all')]
        self.eval_paths = [os.path.join(self.data_root, 'gold', path)
                           for path in ['bosonnlp/auto_comments.txt', 'bosonnlp/food_comments.txt',
                                        'bosonnlp/news.txt', 'bosonnlp/weibo.txt',
                                        'ctb.gold', 'msr_test_gold.utf8',
                                        'pku_test_gold.utf8']]

        self.model_file = 'model/model'

        self.lm_weight = 0.5

        self.eval_step = 100

config = Config()

from .loader import DataLoader

loader = DataLoader(config.train_paths, config.max_vocab_size)

dict, tagger, training_data = loader.get_data(config.train_paths, config.batch_size)

eval_data = list(loader.batch(config.eval_paths, config.batch_size))

import random
random.shuffle(training_data)
random.shuffle(eval_data)

def bisection(data):
    random.shuffle(data)
    return data[:len(data)//2], data[len(data)//2:]

valid_data, eval_data = bisection(eval_data)

model = BiLSTMCRF(len(dict), len(tagger), config.embedding_size, config.hidden_size, config.dropout)
if config.use_cuda:
    model = model.cuda()
    training_data = [(PackedSequence(sentences.data.cuda(), sentences.batch_sizes),
                      PackedSequence(batch_tags.data.cuda(), batch_tags.batch_sizes))
                     for sentences, batch_tags in training_data]
    eval_data = [(PackedSequence(sentences.data.cuda(), sentences.batch_sizes),
                  PackedSequence(batch_tags.data.cuda(), batch_tags.batch_sizes))
                 for sentences, batch_tags in eval_data]

#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

optimizer = torch.optim.Adam(model.parameters())



def unpack(pad_sequence):
    seqs, lens = pad_packed_sequence(pad_sequence, batch_first=True)
    return [seq[:len] for seq, len in zip(seqs, lens)]



def evaluation_one(pred, gold):
    count = 0
    true = 0
    pos = 0

    start = 0
    for curr in range(0, len(pred)):
        if tagger.is_split(pred[curr]):
            pos += 1

        if tagger.is_split(gold[curr]):
            flag = pred[curr] == gold[curr]
            if flag:
                for k in range(start, curr):
                    if pred[k] != gold[k]:
                        flag = False
                        break
                if flag:
                    count += 1
            true += 1
            start = curr + 1

    return count, true, pos


def evaluation(model, data):

    model.eval()
    correct = 0
    true = 0
    pos = 0
    for sentences, gold_tags in data:
        pred = model(sentences)
        gold_tags = unpack(gold_tags)

        #print(gold_tags)
        for pred, gold in zip(pred, gold_tags):
            c, t, p = evaluation_one(pred[1], list(gold.data))
            correct += c
            true += t
            pos += p

    prec = correct/float(pos+1e-5)
    recall = correct/float(true+1e-5)
    return prec, recall, 2*prec*recall/(prec+recall+1e-5)





# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
    crf_loss = 0
    lm_loss = 0
    total_loss = 0

    best_valid_loss = 1e8

    import time
    start_time = time.time()

    for index, (batch_sentence, batch_tags) in enumerate(training_data):
        model.train()
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Run our forward pass.
        batch_crf_loss, batch_lm_loss = model.loss(batch_sentence, batch_tags)


        batch_loss = batch_crf_loss + batch_lm_loss * config.lm_weight

        crf_loss += batch_crf_loss.data[0]
        lm_loss += batch_lm_loss.data[0]
        total_loss += batch_loss.data[0]

        # Step 3. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
        optimizer.step()

        if index % config.eval_step == 0:
            print('train: total loss = %f, crf loss = %f, lm loss = %f, lm ppl = %f, speed = %f' %
                  (total_loss / (config.batch_size * config.eval_step),
                   (crf_loss / (config.batch_size * config.eval_step)),
                   lm_loss / config.eval_step,
                   math.exp(lm_loss / config.eval_step),
                   (time.time() - start_time) / (config.batch_size * config.eval_step)))
            crf_loss = 0
            lm_loss = 0
            total_loss = 0

            model.eval()

            # valid
            valid_crf_loss = 0
            valid_lm_loss = 0
            valid_total_loss = 0
            for sentences, tags in valid_data:
                batch_valid_crf_loss, batch_valid_lm_loss = model.loss(sentences, tags)
                valid_crf_loss += batch_valid_crf_loss.data[0]
                valid_lm_loss += batch_valid_lm_loss.data[0]
                valid_loss = batch_valid_crf_loss + batch_valid_lm_loss * config.lm_weight
                valid_total_loss += valid_loss.data[0]

            print('valid: total loss = %f, crf loss = %f, lm loss = %f, lm ppl = %f' %
                  (valid_total_loss / (config.batch_size * config.eval_step),
                   (valid_crf_loss / (config.batch_size * config.eval_step)),
                   valid_lm_loss / (config.eval_step),
                   math.exp(valid_lm_loss / config.eval_step)))

            if valid_total_loss < best_valid_loss:
                for group in optimizer.param_groups:
                    group['lr'] /= 2
            else:
                best_valid_loss = valid_total_loss

            # evaluation
            prec, recall, f_score = evaluation(model, eval_data)
            print('metrics: eval prec = %f  recall = %f  F-score = %f' % (prec, recall, f_score))

            # sample
            sentences, gold_tags = valid_data[random.randint(0, len(valid_data) - 1)]
            for sentence, pred in zip(unpack(sentences), model(sentences)):
                pred_score, tag_ixs = pred
                print(pred_score)
                seg = ''.join([dict.get_word(word) + ' ' if tagger.is_split(ix) else dict.get_word(word)
                               for word, ix in zip(list(sentence.data), list(tag_ixs))])
                print('dst: %s' % seg)



            start_time = time.time()

    with open('%s.%d' % (config.model_file, epoch), 'wb') as f:
        torch.save(model, f)

# Check predictions after training
# We got it!










