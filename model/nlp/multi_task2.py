
from .crf import CRFLayer
from .loader import Vocab, BMESTagger, CharacterAttribute, Gazetteer, DataLoader
from .config2 import MultiTaskConfig
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import itertools
import random

import torch
import torch.nn as nn
from torch.nn import Embedding
from .qrnn import QRNN
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from .crf import Embedding

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



class Task(nn.Module):
    def __init__(self):
        super(Task, self).__init__()

    def loss(self, encoder, data):
        pass

    def evaluation(self, encoder, data):
        pass

    def sample(self, encoder, data):
        pass


class TaggerTask(Task):
    def __init__(self, name, encoder, vocab, tagger,
                 hidden_dim, window_sizes, bidirectional, dropout,
                 general_weight_decay=1e-6,
                 task_weight_decay=1e-5):
        super(TaggerTask, self).__init__()

        self.name = name
        self.vocab = vocab
        self.tagger = tagger
        self.general_encoder = encoder
        self.task_encoder = QRNN(self.general_encoder.output_dim(),
                                 hidden_dim, 1, window_sizes,
                                 bidirectional, dropout)
        self.crf = CRFLayer(hidden_dim * 2, len(self.tagger), dropout)

        self.params = [{'params': self.general_encoder.parameters(), 'weight_decay': general_weight_decay},
                       {'params': self.task_encoder.parameters(), 'weight_decay': task_weight_decay},
                       {'params': self.crf.parameters(), 'weight_decay': task_weight_decay}]


    def loss(self, data):
        sentences, gazetteers, gold_tags = data
        crf_feature, _ = self.task_encoder(self.general_encoder(sentences, gazetteers))
        return self.crf.neg_log_likelihood(crf_feature, gold_tags)

    def forward(self, data):
        sentences, gazetteers = data
        crf_feature, _ = self.task_encoder(self.general_encoder(sentences, gazetteers))
        return self.crf(crf_feature)

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

        self.eval()
        correct = 0
        true = 0
        pos = 0
        for sentences, gazetteers, gold_tags in data:
            pred = self.forward((sentences, gazetteers))
            gold_tags = self._unpack(gold_tags)

            #print(gold_tags)
            for pred, gold in zip(pred, gold_tags):
                c, t, p = self.evaluation_one(pred[1], list(gold.data))
                correct += c
                true += t
                pos += p

        prec = correct/float(pos+1e-5)
        recall = correct/float(true+1e-5)

        return {'prec':prec, 'recall':recall, 'f-score':2*prec*recall/(prec+recall+1e-5)}


    def sample(self, data):

        self.eval()

        import random
        sentences, gazetteers, gold_tags = data
        for sentence, pred in zip(self._unpack(sentences), self.forward((sentences, gazetteers))):
            pred_score, tag_ixs = pred
            seg = ''.join([self.vocab.get_word(word) + ' ' if self.tagger.is_split(ix) else self.vocab.get_word(word)
                           for word, ix in zip(list(sentence.data), list(tag_ixs))])

            yield '%.6f\t%s' % (pred_score, seg)

class MultiTask:

    def __init__(self, tasks, task_weights, use_cuda=False):
        super(MultiTask, self).__init__()

        self.use_cuda = use_cuda

        self.tasks = [t.cuda() if use_cuda else t for t in tasks]
        self.task_weights = task_weights

        self.optimizers = [torch.optim.Adam(t.params, lr=2e-3) for t in self.tasks]


    @staticmethod
    def to_cuda(batch_data):
        sentences, gazetteers, batch_tags = batch_data
        return (PackedSequence(sentences.data.cuda(), sentences.batch_sizes),
                PackedSequence(gazetteers.data.cuda(), gazetteers.batch_sizes),
                PackedSequence(batch_tags.data.cuda(), batch_tags.batch_sizes))



    def train(self, train_data, valid_data, eval_data, epoch, eval_step, model_prefix):

        # put valid_data, eval_data to gpu.   put train_data to gpu lazily
        if self.use_cuda:
            valid_data = [(task_id, self.to_cuda(batch_data)) for task_id, batch_data in valid_data]
            eval_data = [(task_id, [self.to_cuda(batch) for batch in task_data])
                         for task_id, task_data in eval_data]

        for epoch in tqdm(range(epoch), desc='epoch', total=epoch):
            train_losses = [0.] * len(self.tasks)
            task_step_count = [0] * len(self.tasks)
            for batch_id, (task_id, task_data) in tqdm(enumerate(train_data, start=1),
                                                       desc='batch',
                                                       total=len(train_data)):
                task = self.tasks[task_id]
                task.train()
                task.zero_grad()

                if self.use_cuda:
                    task_data = self.to_cuda(task_data)

                loss = task.loss(task_data)
                loss.backward()
                train_losses[task_id] += loss.data[0]
                task_step_count[task_id] += len(task_data)

                # Step 3. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                torch.nn.utils.clip_grad_norm(task.parameters(), 0.25)
                self.optimizers[task_id].step()

                if batch_id % eval_step == 0:

                    def print_loss(prefix, losses):
                        print(prefix + '\t' + '\t'.join(['%s=%.6f' % (name, loss) for name, loss in losses]))
                    print()
                    print_loss('train loss:', [(task.name, loss/step_count)
                                for task, loss, step_count in zip(self.tasks, train_losses, task_step_count)])
                    train_losses = [0.] * len(self.tasks)
                    task_step_count = [0] * len(self.tasks)

                    valid_losses = self.valid(valid_data)
                    print()
                    print_loss('valid loss:', valid_losses)


                    # sample
                    self.sample(eval_data)

            self.sample(eval_data)
            print('eval:')
            for task_id, task_data in eval_data:
                result = self.tasks[task_id].evaluation(task_data)
                print('%s\t%s' % (self.tasks[task_id].name,
                                  '\t'.join(["%s=%0.6f" % (k, v)for k, v in result.items()])))

            for task in self.tasks:
                with open('%s.%s.%d' % (model_prefix, task.name, epoch), 'wb') as f:
                    state_dict = task.state_dict()
                    for elem in state_dict:
                        state_dict[elem].cpu()
                    torch.save(state_dict, f)

    def valid(self, data):
        losses = [0.] * len(self.tasks)
        counts = [0] * len(self.tasks)
        for task_id, task_data in data:
            task = self.tasks[task_id]
            task.eval()
            losses[task_id] += task.loss(task_data).data[0]
            counts[task_id] += len(task_data)

        return [(task.name, loss/count) for task, loss, count in zip(self.tasks, losses, counts)]

    def sample(self, data):
        print('sample:')
        for task_id, task_data in data:
            if len(task_data) > 0:
                print('task %s' % self.tasks[task_id].name)
                print('\n'.join(self.tasks[task_id].sample(task_data[random.randint(0, len(task_data) - 1)])))


def build(config):
    # build vocab
    word_with_tag = [DataLoader.count(t.train_paths, t.with_type) for t in config.tasks]

    import collections

    word_counts = collections.defaultdict()

    for counts, tags in word_with_tag:
        word_counts.update(counts)

    task_sizes = [sum(counts.values()) for counts, _ in word_with_tag]

    min_sizes = min(task_sizes)
    task_weights = [float(min_sizes)/s for s in task_sizes]

    vocab = Vocab.build(word_counts, config.max_vocab_size)
    taggers = [BMESTagger(tags) for _, tags in word_with_tag]

    char2attr = CharacterAttribute.load(config.char_attr)
    gazetteers = [char2attr] + [Gazetteer.load(name, path) for name, path in config.wordset]

    loaders = [DataLoader(vocab, tagger, gazetteers, task.with_type) for tagger, task in zip(taggers, config.tasks)]

    train_data = []
    valid_data = []
    eval_data = []


    for id, (loader, task) in enumerate(zip(loaders, config.tasks)):
        temp_train, temp_valid = train_test_split(list(loader.get_data(task.train_paths, config.batch_size)),
                                                  test_size=config.test_size)

        train_data += zip([id] * len(temp_train), temp_train)
        valid_data += zip([id] * len(temp_valid), temp_valid)

        temp_eval = list(loader.get_data(task.eval_paths, config.batch_size))
        if len(temp_eval) > 0:
            eval_data.append((id, temp_eval))

    random.shuffle(train_data)
    random.shuffle(valid_data)

    encoder = Encoder(len(vocab), gazetteers, config.embedding_dim,
                      config.hidden_mode, config.hidden_dim, config.num_hidden_layer, config.window_sizes,
                      config.dropout)

    tasks = [TaggerTask(task.name, encoder, vocab, tagger,
                        task.hidden_dim, task.window_sizes, task.bidirectional, task.dropout)
             for id, (tagger, task) in enumerate(zip(taggers, config.tasks))]

    return tasks, task_weights, (train_data, valid_data, eval_data)

config = MultiTaskConfig()

tasks, task_weights, (train_data, valid_data, eval_data) = build(config)

trainer = MultiTask(tasks, task_weights, config.use_cuda)
print('train stage')
trainer.train(train_data, valid_data, eval_data, config.epoches, config.eval_step, config.model_prefix)