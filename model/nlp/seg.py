
import os
import math
import itertools
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

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


        if gazetteers is not None:
            self.gazetteers = gazetteers
            self.gazetteers_embeddings = [nn.Linear(gazetteer.length(), embedding_dim)
                               for gazetteer in gazetteers]
            self.gazetteers_len = [gazetteer.length() for gazetteer in gazetteers]

            self.gazetteers_index = [0] + list(itertools.accumulate(self.gazetteers_len))[0:-1]

    def forward(self, sentence, gazetteers):
        '''
        :param input: PackedSequence
        :return:
        '''
        sentence, batch_sizes = sentence

        word_emb = self.word_embedding(sentence)

        if gazetteers is not None and len(self.gazetteers) > 0:
            gazetteers, batch_sizes = gazetteers

            outputs = [embedding(gazetteers[:, start:start+length]).unsqueeze(-1)
                       for embedding, (start, length) in zip(self.gazetteers_embeddings,
                                                             zip(self.gazetteers_index, self.gazetteers_len))]

        output = word_emb + torch.cat(outputs, -1).sum(-1)

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
                context = hidden_states[context_start:context_start+batch_sizes[i], 0:self.dim]
                next = sentences[next_start:next_start+batch_sizes[i]]
                output = self.output_embedding(self.forward_linear(context))
                loss += self.cross_entropy(output, next)
                context_start += batch_sizes[i-1]
                next_start += batch_sizes[i]
            

            if self.bidirectional:
                # backward language model
                context_start = total
                next_start = total - batch_sizes[-1]
                for i in range(len(batch_sizes)-2, -1, -1):
                    context_start -= batch_sizes[i+1]
                    next_start -= batch_sizes[i]
                    context = hidden_states[context_start:context_start+batch_sizes[i+1], self.dim:]
                    next = sentences[next_start:next_start+batch_sizes[i+1]]
                    output = self.output_embedding(self.backward_linear(context))
                    loss += self.cross_entropy(output, next)

                denom += total - batch_sizes[0]


            return loss/denom
        else:
            return loss


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, num_label, gazetteers, embedding_dim, hidden_mode, num_hidden_layer=1, dropout=0.5):
        super(BiLSTMCRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.input_embed = Embedding(self.word_embeds, self.embedding_dim, gazetteers)

        self.hidden_dim = self.embedding_dim
        self.num_hidden_layer = num_hidden_layer
        self.num_direction = 2

        self.hidden_mode = hidden_mode

        if self.hidden_mode == 'QRNN':
            from .qrnn import QRNN
            self.hidden_module = QRNN(self.embedding_dim, self.hidden_dim, self.num_hidden_layer,
                                      kernel_size=5, dropout = config.dropout)
        else:
            self.hidden_module = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_hidden_layer,
                                         bidirectional=True, dropout=dropout)

        # output layer
        self.crf = CRFLayer(embedding_dim * 2, num_label, dropout)
        self.lm = LanguageModel(embedding_dim, vocab_size, self.word_embeds.weight, bidirectional=True, dropout=dropout)


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
        return self.crf.neg_log_likelihood(feats, tags), self.lm.criterion(sentence, feats)



torch.set_num_threads(10)

class Config:
    def __init__(self):
        self.max_vocab_size = 5000
        self.batch_size = 32
        self.embedding_size = 128
        self.hidden_mode = 'QRNN'
        self.num_hidden_layer = 2
        self.dropout = 0.3
        self.use_cuda = True

        self.data_root = '/Users/sunqf/startup/quotesbot/nlp-data/chinese_segment/data/'
        #self.data_root = '/home/sunqf/Work/chinese_segment/data'
        self.train_paths = [os.path.join(self.data_root, 'train/train.all')]
        self.eval_paths = [os.path.join(self.data_root, 'gold', path)
                           for path in ['bosonnlp/auto_comments.txt', 'bosonnlp/food_comments.txt',
                                        'bosonnlp/news.txt', 'bosonnlp/weibo.txt',
                                        'ctb.gold', 'msr_test_gold.utf8',
                                        'pku_test_gold.utf8']]

        # 字属性字典
        self.char_attr = './dict/word-type'
        # 词集合
        self.wordset = {}
        self.model_file = 'model/model'

        self.lm_weight = 0.5

        self.eval_step = 2000


config = Config()

from .loader import DataLoader

loader = DataLoader(config.train_paths, config.char_attr, config.wordset, config.max_vocab_size)

vocab, gazetteers, tagger, training_data = loader.get_data(config.train_paths, config.batch_size)

eval_data = list(loader.batch(config.eval_paths, config.batch_size))

import random
random.shuffle(training_data)
random.shuffle(eval_data)

valid_data, eval_data = train_test_split(eval_data, test_size=0.7)

model = BiLSTMCRF(len(vocab), len(tagger), gazetteers, config.embedding_size,
                  config.hidden_mode, config.num_hidden_layer, config.dropout)

if config.use_cuda:
    model = model.cuda()
    training_data = [(PackedSequence(sentences.data.cuda(), sentences.batch_sizes),
                      PackedSequence(gazetteers.data.cuda(), gazetteers.batch_sizes),
                      PackedSequence(batch_tags.data.cuda(), batch_tags.batch_sizes))
                     for sentences, gazetteers, batch_tags in training_data]
    valid_data = [(PackedSequence(sentences.data.cuda(), sentences.batch_sizes),
                   PackedSequence(gazetteers.data.cuda(), gazetteers.batch_sizes),
                   PackedSequence(batch_tags.data.cuda(), batch_tags.batch_sizes))
                  for sentences, gazetteers, batch_tags in valid_data]
    eval_data = [(PackedSequence(sentences.data.cuda(), sentences.batch_sizes),
                  PackedSequence(gazetteers.data.cuda(), gazetteers.batch_sizes),
                  PackedSequence(batch_tags.data.cuda(), batch_tags.batch_sizes))
                 for sentences, gazetteers, batch_tags in eval_data]

print(model)

#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

optimizer = torch.optim.Adam(model.parameters())
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)


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
    for sentences, gazetteers, gold_tags in data:
        pred = model(sentences, gazetteers)
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

    best_valid_loss = None

    scheduler.step()

    import time
    start_time = time.time()

    for index, (batch_sentence, batch_gazetteers, batch_tags) in enumerate(training_data):
        model.train()
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Run our forward pass.
        batch_crf_loss, batch_lm_loss = model.loss(batch_sentence, batch_gazetteers, batch_tags)


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
            valid_len = len(valid_data)
            for sentences, gazetteers, tags in valid_data:
                batch_valid_crf_loss, batch_valid_lm_loss = model.loss(sentences, gazetteers, tags)
                valid_crf_loss += batch_valid_crf_loss.data[0]
                valid_lm_loss += batch_valid_lm_loss.data[0]
                valid_loss = batch_valid_crf_loss + batch_valid_lm_loss * config.lm_weight
                valid_total_loss += valid_loss.data[0]

            print('valid: total loss = %f, crf loss = %f, lm loss = %f, lm ppl = %f' %
                  (valid_total_loss / (config.batch_size * valid_len),
                   valid_crf_loss / (config.batch_size * valid_len),
                   valid_lm_loss / valid_len,
                   math.exp(valid_lm_loss / valid_len)))
            '''
            if (best_valid_loss is None) or (valid_total_loss < best_valid_loss):
                best_valid_loss = valid_total_loss
            else:
                for group in optimizer.param_groups:
                    group['lr'] /= 2
            '''

            # sample
            sentences, gazetteers, gold_tags = valid_data[random.randint(0, len(valid_data) - 1)]
            for sentence, pred in zip(unpack(sentences), model(sentences, gazetteers)):
                pred_score, tag_ixs = pred
                print(pred_score)
                seg = ''.join([vocab.get_word(word) + ' ' if tagger.is_split(ix) else vocab.get_word(word)
                               for word, ix in zip(list(sentence.data), list(tag_ixs))])
                print('dst: %s' % seg)
            start_time = time.time()


    # evaluation
    prec, recall, f_score = evaluation(model, eval_data)
    print('metrics: eval prec = %f  recall = %f  F-score = %f' % (prec, recall, f_score))


    with open('%s.opt.%d' % (config.model_file, epoch), 'wb') as f:
        torch.save(model, f)

# Check predictions after training
# We got it!











