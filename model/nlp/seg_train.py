
import os
import math
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence

from torch import nn

from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

import torch
from .crf import BiLSTMCRF
from .config import Config

from .loader import DataLoader
import pickle

import random


class SegTrainer:

    def __init__(self, config=Config(), coarse=True, fine=True):
        self.config = config
        self.loader = DataLoader(config.coarse_train_paths, config.char_attr, config.wordset, config.max_vocab_size)

        self.coarse = coarse
        self.fine = fine

    def _coarse_init(self):
        self.vocab, self.gazetteers, self.tagger, self.training_data = self.loader.get_data(self.config.coarse_train_paths, self.config.batch_size)

        eval_data = list(self.loader.batch(self.config.coarse_eval_paths, self.config.batch_size))

        import random
        random.shuffle(self.training_data)
        random.shuffle(eval_data)

        self.valid_data, self.eval_data = train_test_split(eval_data, test_size=0.7)

        self.model = BiLSTMCRF(len(self.vocab), len(self.tagger), self.gazetteers, self.config.embedding_dim,
                               self.config.hidden_mode, self.config.hidden_dim, self.config.num_hidden_layer,
                               self.config.window_sizes, self.config.dropout)

        self._to_cuda()

    def _fine_init(self):
        self.vocab, self.gazetteers, self.tagger, self.coarse_training_data = self.loader.get_data(self.config.fine_train_paths, self.config.batch_size)

        eval_data = list(self.loader.batch(self.config.fine_eval_paths, self.config.batch_size))

        import random
        random.shuffle(self.training_data)
        random.shuffle(eval_data)

        self.valid_data, self.eval_data = train_test_split(eval_data, test_size=0.7)

        if hasattr(self, 'model') is False:
            with open('%s.coarse.%d' % (self.config.model_prefix, self.config.epoch - 1)) as file:
                self.model = torch.load(file)

        self._to_cuda()

    def _to_cuda(self):
        if self.config.use_cuda:
            self.model = self.model.cuda()
            self.training_data = [(PackedSequence(sentences.data.cuda(), sentences.batch_sizes),
                                   PackedSequence(gazetteers.data.cuda(), gazetteers.batch_sizes),
                                   PackedSequence(batch_tags.data.cuda(), batch_tags.batch_sizes))
                                  for sentences, gazetteers, batch_tags in self.training_data]
            self.valid_data = [(PackedSequence(sentences.data.cuda(), sentences.batch_sizes),
                                PackedSequence(gazetteers.data.cuda(), gazetteers.batch_sizes),
                                PackedSequence(batch_tags.data.cuda(), batch_tags.batch_sizes))
                               for sentences, gazetteers, batch_tags in self.valid_data]
            self.eval_data = [(PackedSequence(sentences.data.cuda(), sentences.batch_sizes),
                               PackedSequence(gazetteers.data.cuda(), gazetteers.batch_sizes),
                               PackedSequence(batch_tags.data.cuda(), batch_tags.batch_sizes))
                              for sentences, gazetteers, batch_tags in self.eval_data]

    def _unpack(self, pad_sequence):
        seqs, lens = pad_packed_sequence(pad_sequence, batch_first=True)
        return [seq[:len] for seq, len in zip(seqs, lens)]

    def evaluation_one(self, pred, gold):
        count = 0
        true = 0
        pos = 0

        start = 0
        for curr in range(0, len(pred)):
            if self.tagger.is_split(pred[curr]):
                pos += 1

            if self.tagger.is_split(gold[curr]):
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


    def evaluation(self, data):
        self.model.eval()
        correct = 0
        true = 0
        pos = 0
        for sentences, gazetteers, gold_tags in data:
            pred = self.model(sentences, gazetteers)
            gold_tags = self._unpack(gold_tags)

            #print(gold_tags)
            for pred, gold in zip(pred, gold_tags):
                c, t, p = self.evaluation_one(pred[1], list(gold.data))
                correct += c
                true += t
                pos += p

        prec = correct/float(pos+1e-5)
        recall = correct/float(true+1e-5)
        return prec, recall, 2*prec*recall/(prec+recall+1e-5)

    def train(self):

        if self.coarse:
            self._coarse_init()
            self.optimizer = torch.optim.Adam(self.model.parameters())
            self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)

            log_prefix = 'coarse phase'
            # Make sure prepare_sequence from earlier in the LSTM section is loaded
            for epoch in range(self.config.coarse_epoches):
                self.scheduler.step()

                self.train_one(log_prefix)

                # evaluation
                prec, recall, f_score = self.evaluation(self.eval_data)
                print(log_prefix + ' metrics: eval prec = %f  recall = %f  F-score = %f' % (prec, recall, f_score))

                with open('%s.coarse.%d' % (self.config.model_prefix, epoch), 'wb') as f:
                    torch.save(self.model, f)

        if self.fine:
            self._fine_init()
            self.optimizer = torch.optim.Adam(self.model.parameters())
            self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)

            log_prefix = 'fine phase'
            # Make sure prepare_sequence from earlier in the LSTM section is loaded
            for epoch in range(self.config.fine_epoches):
                self.scheduler.step()

                self.train_one(log_prefix)

                # evaluation
                prec, recall, f_score = self.evaluation(self.eval_data)
                print(log_prefix + ' metrics: eval prec = %f  recall = %f  F-score = %f' % (prec, recall, f_score))

                with open('%s.fine.%d' % (self.config.model_prefix, epoch), 'wb') as f:
                    torch.save(self.model, f)

    def train_one(self, log_prefix):
        import time
        start_time = time.time()

        total_loss = 0

        best_valid_loss = None

        for index, (batch_sentence, batch_gazetteers, batch_tags) in enumerate(self.training_data, start=1):
            self.model.train()
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            self.model.zero_grad()

            # Step 2. Run our forward pass.
            batch_loss = self.model.loss(batch_sentence, batch_gazetteers, batch_tags)

            total_loss += batch_loss.data[0]

            # Step 3. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.25)
            self.optimizer.step()

            if index % self.config.eval_step == 0:
                print(log_prefix + ' train: loss = %f, speed = %f' %
                      (total_loss / (self.config.batch_size * self.config.eval_step),
                       (time.time() - start_time) / (self.config.batch_size * self.config.eval_step)))
                crf_loss = 0
                lm_loss = 0
                total_loss = 0

                self.model.eval()

                # valid
                valid_crf_loss = 0
                valid_lm_loss = 0
                valid_total_loss = 0
                valid_len = len(self.valid_data)
                for sentences, gazetteers, tags in self.valid_data:
                    #batch_valid_crf_loss, batch_valid_lm_loss = model.loss(sentences, gazetteers, tags)
                    batch_valid_crf_loss = self.model.loss(sentences, gazetteers, tags)

                    valid_crf_loss += batch_valid_crf_loss.data[0]
                    #valid_lm_loss += batch_valid_lm_loss.data[0]
                    valid_loss = batch_valid_crf_loss #+ batch_valid_lm_loss * config.lm_weight
                    valid_total_loss += valid_loss.data[0]

                print(log_prefix + ' valid: loss = %f' % (valid_total_loss / (self.config.batch_size * valid_len)))
                '''
                if (best_valid_loss is None) or (valid_total_loss < best_valid_loss):
                    best_valid_loss = valid_total_loss
                else:
                    for group in optimizer.param_groups:
                        group['lr'] /= 2
                '''

                # sample
                sentences, gazetteers, gold_tags = self.valid_data[random.randint(0, len(self.valid_data) - 1)]
                for sentence, pred in zip(self._unpack(sentences), self.model(sentences, gazetteers)):
                    pred_score, tag_ixs = pred
                    print(pred_score)
                    seg = ''.join([self.vocab.get_word(word) + ' ' if self.tagger.is_split(ix) else self.vocab.get_word(word)
                                   for word, ix in zip(list(sentence.data), list(tag_ixs))])
                    print(log_prefix + ' dst: %s' % seg)
                start_time = time.time()

torch.set_num_threads(10)

trainer = SegTrainer(Config(), True, True)

trainer.train()














