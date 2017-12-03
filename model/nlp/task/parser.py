import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from collections import defaultdict, namedtuple
import itertools
from ..layer.encoder import Encoder
from ..layer.qrnn import QRNN
from ..layer.crf import CRFLayer
from .task import Task, Loader
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence
from ..util.vocab import Vocab, CharacterAttribute, Gazetteer
from sklearn.model_selection import train_test_split

import numpy as np

def tree_lstm(c1, c2, lstm_in):
    a, i, f1, f2, o = lstm_in.chunk(5, 1)
    c = a.tanh() * i.sigmoid() + f1.sigmoid() * c1 + f2.sigmoid() * c2
    h = o.sigmoid() * c.tanh()
    return h, c

class BinaryTreeLSTM(nn.Module):

    def __init__(self, size, tracker_size):
        super(BinaryTreeLSTM, self).__init__()

        self.left = nn.Linear(size, 5 * size)
        self.right = nn.Linear(size, 5 * size, bias=False)
        if tracker_size is not None:
            self.track = nn.Linear(tracker_size, 5 * size, bias=False)

    def forward(self, left, right, tracking):
        lstm_in = self.left(left[0])
        lstm_in += self.right(right[0])
        if hasattr(self, 'track'):
            lstm_in += self.track(tracking[0])

        return tree_lstm(left[1], right[1], lstm_in)

class DependencyTreeLSTM(nn.Module):

    def __init__(self, size, tracker_size):
        super(DependencyTreeLSTM, self).__init__()

        self.size = size
        # input, output, update
        self.iou = nn.Linear(size, 3 * size)

        # forget
        self.forget = nn.Linear(size, size)

        self.tracker_size = tracker_size
        if tracker_size is not None:
            self.iou_track = nn.Linear(self.tracker_size, 3 * size, bias=False)
            self.forget_track = nn.Linear(self.tracker_size, size, bias=False)

    def forward(self, childrens, tracking):
        '''
        :param children: [batch * [num_child * Tensor(1, size)]
        :param tracking: [batch * Tensor[1, size]]
        :return: [batch * Tensor[1, size]]
        '''
        lens = [len(children) for children in childrens]

        tracking_h, tracking_c = torch.cat(tracking, 0).chunk(2, 1)

        childrens = [torch.cat(children, 0) for children in childrens]

        mean_h, mean_c = torch.cat([children.mean(0) for children in childrens], 0).chunk(2, 1)
        iou = self.iou(mean_h)
        if self.tracker_size:
            iou += self.iou_track(tracking_h)

        i, o, u = iou.chunk(3, 1)

        i = i.sigmoid()
        o = o.sigmoid()
        u = u.tanh()

        children_h, children_c = torch.cat(childrens, 0).chunk(2, 1)
        f = self.forget(children_h)
        if self.tracker_size:
            forget_track = self.forget_track(tracking_h)

            forget_track = torch.cat([forget_track[b].expand(len, self.size)
                                      for b, len in enumerate(lens)],
                                     0)
            f += forget_track


        fc = f.sigmoid() * children_c

        cumfc = []
        start = 0
        for len in lens:
            cumfc.append(fc[0:len].sum(0).unsqueeze(0))
            start += len

        fc = torch.cat(cumfc, 0)

        c = i * u + fc
        h = o * c

        return unbundle((h, c))

def bundle(lstm_iter):
    if lstm_iter is None:
        return None
    lstm_iter = tuple(lstm_iter)
    if lstm_iter[0] is None:
        return None
    return torch.cat(lstm_iter, 0).chunk(2, 1)


def unbundle(state):
    if state is None:
        return itertools.repeat(None)
    return torch.split(torch.cat(state, 1), 1, 0)


class Reduce(nn.Module):
    """TreeLSTM composition module for SPINN.
    The TreeLSTM has two or three inputs: the first two are the left and right
    children being composed; the third is the current state of the tracker
    LSTM if one is present in the SPINN model.
    Args:
        size: The size of the model state.
        tracker_size: The size of the tracker LSTM hidden state, or None if no
            tracker is present.
    """

    def __init__(self, size, tracker_size=None):
        super(Reduce, self).__init__()
        self.tree_lstm = BinaryTreeLSTM(size, tracker_size)

    def forward(self, left_in, right_in, tracking=None):
        """Perform batched TreeLSTM composition.
        This implements the REDUCE operation of a SPINN in parallel for a
        batch of nodes. The batch size is flexible; only provide this function
        the nodes that actually need to be REDUCEd.
        The TreeLSTM has two or three inputs: the first two are the left and
        right children being composed; the third is the current state of the
        tracker LSTM if one is present in the SPINN model. All are provided as
        iterables and batched internally into tensors.
        Additionally augments each new node with pointers to its children.
        Args:
            left_in: Iterable of ``B`` ~autograd.Variable objects containing
                ``c`` and ``h`` concatenated for the left child of each node
                in the batch.
            right_in: Iterable of ``B`` ~autograd.Variable objects containing
                ``c`` and ``h`` concatenated for the right child of each node
                in the batch.
            tracking: Iterable of ``B`` ~autograd.Variable objects containing
                ``c`` and ``h`` concatenated for the tracker LSTM state of
                each node in the batch, or None.
        Returns:
            out: Tuple of ``B`` ~autograd.Variable objects containing ``c`` and
                ``h`` concatenated for the LSTM state of each new node. These
                objects are also augmented with ``left`` and ``right``
                attributes.
        """
        left, right = bundle(left_in), bundle(right_in)
        tracking = bundle(tracking)
        out = unbundle(self.tree_lstm(left, right, tracking))
        # for o, l, r in zip(out, left_in, right_in):
        #     o.left, o.right = l, r
        return out

class Contexter(nn.Module):

    def __init__(self, size, tracker_size, predict):
        super(Contexter, self).__init__()
        self.rnn = nn.LSTMCell(3 * size, tracker_size)
        self.state_size = tracker_size
        if predict:
            self.transition = nn.Linear(tracker_size, 3)

    def forward(self, bufs, stacks, prev_state):
        '''

        :param bufs: buffer list.
        :param stacks: stack list.
        :param prev:
        :return:
        '''
        buf = bundle(buf[-1] for buf in bufs)[0]
        stack1 = bundle(stack[-1] for stack in stacks)[0]
        stack2 = bundle(stack[-2] for stack in stacks)[0]

        x = torch.cat((buf, stack1, stack2), 1)

        prev = bundle(s[-1] for s in prev_state)
        prev = self.rnn(x, prev)
        if hasattr(self, 'transition'):
            return unbundle(prev), self.transition(prev[0])

        return unbundle(prev)

class Action:
    NONE = 0
    SHIFT = 1
    REDUCE = 2

    def convert(action):
        if action == 'shift':
            return Action.SHIFT
        elif action == 'reduce':
            return Action.REDUCE
        else:
            return Action.NONE

class SPINN(nn.Module):

    def __init__(self, input_dim, feature_dim, tracker_dim, predict):
        super(SPINN, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.tracker_dim = tracker_dim
        self.projection = nn.Linear(input_dim, feature_dim * 2)
        self.reduce = Reduce(feature_dim, tracker_dim)
        self.contexter = Contexter(feature_dim, tracker_dim,
                                   predict=True)

        self.sentence_end = nn.Parameter(torch.FloatTensor(1, feature_dim * 2).zero_())
        self.stack_base = nn.Parameter(torch.FloatTensor(1, feature_dim * 2).zero_())
        self.context_begin = nn.Parameter(torch.FloatTensor(1, tracker_dim * 2).zero_())

        self.reset_parameters()

    def reset_parameters(self):
        import math
        torch.nn.init.xavier_uniform(self.sentence_end)
        torch.nn.init.xavier_uniform(self.stack_base)
        torch.nn.init.xavier_uniform(self.context_begin)

    def _buffer(self, sentences):
        if isinstance(sentences, PackedSequence):
            buffers, batch_sizes = sentences
            buffers = self.projection(buffers)
            buffers, lengths = pad_packed_sequence(PackedSequence(buffers, batch_sizes), batch_first=False)
        else:
            raise 'buffers must be PackedSequence'

        buffers = [list(torch.split(b.squeeze(1)[:length], 1, 0))
                   for b, length in zip(torch.split(buffers, 1, 1), lengths)]

        for b in buffers:
            b.append(self.sentence_end)

        buffers = [list(reversed(b)) for b in buffers]

        return buffers, lengths


    def loss(self, sentences, gold_actions):
        '''
        :param sentences: [length, batch, dim]
        :param gold_actions: [length, batch]
        :return:
        '''

        State = namedtuple('State', ['buffer', 'stack', 'contexts', 'gold_actions'])

        buffers, lengths = self._buffer(sentences)

        # [batch_size * [stack_size * [node]]]
        stacks = [[self.stack_base, self.stack_base]] * len(lengths)
        contexts = [[self.context_begin]] * len(lengths)
        num_action = gold_actions.size(0)

        trans_loss = Variable(torch.FloatTensor([0]))
        trans_correct = 0
        trans_count = 0


        for t in range(num_action):
            buffer_t, stack_t, action_t, context_t = [], [], [], []
            for buf, stack, action, context in zip(buffers, stacks, gold_actions[t].data, contexts):
                if action == Action.SHIFT or action == Action.REDUCE:
                    buffer_t.append(buf)
                    stack_t.append(stack)
                    action_t.append(action)
                    context_t.append(context)

            context_t_output, trans_hyp = self.contexter(buffer_t, stack_t, context_t)

            action_t = Variable(torch.LongTensor(action_t), requires_grad=False)
            trans_loss += F.cross_entropy(trans_hyp,
                                          action_t,
                                          size_average=False,
                                          ignore_index=Action.NONE)
            trans_pred = trans_hyp.max(1)[1]
            trans_correct += (trans_pred.data == action_t.data).sum()
            trans_count += action_t.nelement()

            reducing_lefts, reducing_rights, reducing_stacks, reducing_contexts = [], [], [], []
            for buf, stack, action, context, context_output in zip(buffer_t, stack_t, action_t.data, context_t, context_t_output):
                context.append(context_output)
                if action == Action.SHIFT:
                    stack.append(buf.pop())
                elif action == Action.REDUCE:
                    reducing_rights.append(stack.pop())
                    reducing_lefts.append(stack.pop())
                    reducing_contexts.append(context_output)
                    reducing_stacks.append(stack)


            if reducing_rights:
                reduceds = iter(self.reduce(reducing_lefts, reducing_rights, reducing_contexts))
                for reduced, stack in zip(reduceds, reducing_stacks):
                    stack.append(reduced)

        return bundle([stack.pop() for stack in stacks])[0], trans_loss/trans_count, trans_correct, trans_count


    def range_loss(self, sentences, gold_actions, begin, end):
        '''
        :param sentences: [length, batch, dim]
        :param gold_actions: [length, batch]
        :return:
        '''

        State = namedtuple('State', ['buffer', 'stack', 'contexts', 'gold_actions'])

        buffers, lengths = self._buffer(sentences)

        # [batch_size * [stack_size * [node]]]
        stacks = [[self.stack_base, self.stack_base]] * len(lengths)
        contexts = [[self.context_begin]] * len(lengths)
        num_action = gold_actions.size(0)

        trans_loss = Variable(torch.FloatTensor([0]))
        trans_correct = 0
        trans_count = 0


        for t in range(min(num_action, end)):
            buffer_t, stack_t, action_t, context_t = [], [], [], []
            for buf, stack, action, context in zip(buffers, stacks, gold_actions[t].data, contexts):
                if action == Action.SHIFT or action == Action.REDUCE:
                    buffer_t.append(buf)
                    stack_t.append(stack)
                    action_t.append(action)
                    context_t.append(context)

            context_t_output, trans_hyp = self.contexter(buffer_t, stack_t, context_t)

            action_t = Variable(torch.LongTensor(action_t), requires_grad=False)

            if t >= begin:
                trans_loss += F.cross_entropy(trans_hyp,
                                              action_t,
                                              size_average=False,
                                              ignore_index=Action.NONE)
                trans_pred = trans_hyp.max(1)[1]
                trans_correct += (trans_pred.data == action_t.data).sum()
                trans_count += action_t.nelement()

            reducing_lefts, reducing_rights, reducing_stacks, reducing_contexts = [], [], [], []
            for buf, stack, action, context, context_output in zip(buffer_t, stack_t, action_t.data, context_t, context_t_output):
                context.append(context_output)
                if action == Action.SHIFT:
                    stack.append(buf.pop())
                elif action == Action.REDUCE:
                    reducing_rights.append(stack.pop())
                    reducing_lefts.append(stack.pop())
                    reducing_contexts.append(context_output)
                    reducing_stacks.append(stack)


            if reducing_rights:
                reduceds = iter(self.reduce(reducing_lefts, reducing_rights, reducing_contexts))
                for reduced, stack in zip(reduceds, reducing_stacks):
                    stack.append(reduced)

        return trans_loss/trans_count, trans_correct, trans_count

    def check(self, actions, next, sentence_length):

        shift_count = sum(1 for a in actions if a == Action.SHIFT)
        reduce_count = sum(1 for a in actions if a == Action.REDUCE)

        if next == Action.SHIFT: shift_count += 1
        elif next == Action.REDUCE: reduce_count += 1

        return shift_count <= sentence_length and reduce_count < shift_count

    def beam_parse(self, sentence, beam_size=10):

        ParseState = namedtuple('ParseState', ['buffer', 'stack', 'contexts', 'actions', 'score'])
        sen_len = sentence.size(0)
        max_action = sen_len * 2 - 1
        buffer = list(self.projection(sentence).split(1, 0))
        buffer.append(self.sentence_end)
        topk = [ParseState(list(reversed(buffer)), [self.stack_base, self.stack_base], [self.context_begin], [], 0.0)]

        for t in range(max_action):
            #print([(len(state.buffer), len(state.stack), len(state.contexts), len(state.actions), state.score) for state in topk])
            buffer_t = [b.buffer for b in topk]
            stack_t = [b.stack for b in topk]
            context_t = [b.contexts for b in topk]
            action_t = [b.actions for b in topk]

            context_output, action_hyp = self.contexter(buffer_t, stack_t, context_t)

            action_prob = F.log_softmax(action_hyp).split(1, 0)

            all = [(state, context, action, state.score+probs[0, action].data[0])
                   for action in [Action.SHIFT, Action.REDUCE] for state, context, probs in zip(topk, context_output, action_prob)]

            sort_res = sorted(all, key=lambda i: i[3], reverse=True)

            topk = []
            reducing_left, reducing_right, reducing_context, reducing_stack = [], [], [], []
            for state, context, action, score in sort_res:
                if self.check(state.actions, action, sen_len):
                    buffer = state.buffer.copy()
                    stack = state.stack.copy()
                    contexts = state.contexts.copy()
                    actions = state.actions.copy()
                    contexts.append(context)
                    actions.append(action)

                    if action == Action.SHIFT:
                        stack.append(buffer.pop())
                    elif action == Action.REDUCE:
                        reducing_right.append(stack.pop())
                        reducing_left.append(stack.pop())
                        reducing_context.append(contexts[-1])
                        reducing_stack.append(stack)

                    topk.append(ParseState(buffer, stack, contexts, actions, score))
                    if len(topk) > beam_size:
                        break

            if reducing_right:
                reduceds = self.reduce(reducing_left, reducing_right, reducing_context)
                for reduced, stack in zip(reduceds, reducing_stack):
                    stack.append(reduced)

        return topk[0].actions

class Config:
    def __init__(self):
        self.d_hidden = 128
        self.d_proj = 256
        self.d_tracker = 128
        self.predict = True


class ParserTask(Task):
    def __init__(self, name, encoder, vocab,
                 hidden_dim,
                 tracker_dim,
                 dropout = 0.2,
                 general_weight_decay=1e-6,
                 task_weight_decay=1e-6):
        super(ParserTask, self).__init__()

        self.name = name
        self.vocab = vocab
        self.general_encoder = encoder
        self.tracker_dim = tracker_dim
        self.dropout = dropout
        self.spinn = SPINN(hidden_dim * 2, hidden_dim, tracker_dim, dropout)

        self.params = [{'params': self.general_encoder.parameters(), 'weight_decay': general_weight_decay},
                       #{'params': self.task_encoder.parameters(), 'weight_decay': task_weight_decay},
                       {'params': self.spinn.parameters(), 'weight_decay': task_weight_decay}]

    def forward(self, sentences, actions):
        sentences, gazetteers = sentences

        feature = self.general_encoder(sentences, gazetteers)
        return self.spinn.loss(feature, actions)

    def loss(self, sentences, actions):

        root, loss, acc, count = self.forward(sentences, actions)

        return loss, acc, count

    def range_loss(self, sentences, actions, begin, end):
        sentences, gazetteers = sentences
        feature = self.general_encoder(sentences, gazetteers)
        return self.spinn.range_loss(feature, actions, begin, end)

    def parse(self, sentences):
        sentences, gazetteers = sentences

        feature = self.general_encoder(sentences, gazetteers)

        features, lengths = pad_packed_sequence(feature, batch_first=True)

        return [self.spinn.beam_parse(sentence[0,:length]) for sentence, length in zip(features.split(1, 0), lengths)]

    @staticmethod
    def tree2str(words, actions):
        class Node:
            def __init__(self, value, left=None, right=None):
                self.value = value
                self.left = left
                self.right = right

        stack = []

        wi = 0
        for a in actions:
            if a == Action.SHIFT:
                stack.append(Node(words[wi]))
                wi += 1
            elif a == Action.REDUCE:
                right = stack.pop()
                left = stack.pop()
                stack.append(Node(None, left, right))

        root = stack[-1]

        def dfs(node):
            if node.value:
                return node.value
            else:
                return '(' + dfs(node.left) + ' ' + dfs(node.right) + ')'

        return dfs(root)

    def sample(self, sentences, vocab):

        batch_actions = task.parse(sentences)
        sentences, lengths = pad_packed_sequence(sentences[0], batch_first=True)
        sentences = [sentence[0, :length] for sentence, length in zip(sentences.split(1, 0), lengths)]
        return [self.tree2str(vocab.get_word(sentence.data.numpy()), actions)
                for sentence, actions in zip(sentences, batch_actions)]



class CTBParser(Loader):
    def __init__(self, train_paths, test_paths):
        self.train_data, self.word_counts, self.test_data = self.load(train_paths, test_paths)

        self.train_data = sorted(self.train_data, key=lambda item: len(item[0]), reverse=True)
        self.test_data = sorted(self.test_data, key=lambda item: len(item[0]), reverse=True )

    def load(self, train_paths, test_paths):
        train_data = []
        for path in train_paths:
            with open(path) as file:

                for line in file:
                    line = line.strip()
                    if len(line) > 0:
                        words, transitions = line.split('\t\t')
                        if len(words) > 0:
                            words = words.split()
                            transitions = transitions.split()
                            train_data.append((words, transitions))

                word_counts = defaultdict(int)
                for words, _ in train_data:
                    for w in words:
                        word_counts[w] += 1

        test_data = []
        for path in test_paths:
            with open(path) as file:
                for line in file:
                    line = line.strip()
                    if len(line) > 0:
                        words, transitions = line.split('\t\t')
                        if len(words) > 0:
                            words = words.split()
                            transitions = transitions.split()
                            test_data.append((words, transitions))

        return train_data, word_counts, test_data

    def _batch(self, data, vocab, gazetteers, batch_size):

        gazetteers_dim = sum([c.length() for c in gazetteers])

        for begin in range(0, len(data), batch_size):
            batch = data[begin:begin+batch_size]

            batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)
            sen_lens = [len(s) for s, _ in batch]
            max_sen_len = max(sen_lens)
            max_tran_len = max([len(t) for _, t in batch])
            sentences = torch.LongTensor(max_sen_len, len(batch)).fill_(0)
            gazes = torch.FloatTensor(max_sen_len, len(batch), gazetteers_dim).fill_(0)
            actions = torch.LongTensor(max_tran_len, len(batch)).fill_(0)
            for id, (words, trans) in enumerate(batch):
                sen_len = len(words)
                sentences[:sen_len, id] = torch.from_numpy(vocab.convert(words))
                gazes[0:sen_len, id] = torch.from_numpy(
                    np.concatenate([gazetteer.convert(words) for gazetteer in gazetteers], -1))
                tran_len = len(trans)
                actions[:tran_len, id] = torch.LongTensor([Action.convert(t) for t in trans])
            yield ((pack_padded_sequence(Variable(sentences), sen_lens),
                    pack_padded_sequence(Variable(gazes), sen_lens)),
                    Variable(actions))

    def batch_train(self, vocab, gazetteers, batch_size):
        return self._batch(self.train_data, vocab, gazetteers, batch_size)

    def batch_test(self, vocab, gazetteers, batch_size):
        return self._batch(self.test_data, vocab, gazetteers, batch_size)

class ParserConfig:

    def __init__(self, name, train_paths, test_paths):
        self.name = name
        self.train_paths = train_paths
        self.test_paths = test_paths
        self.hidden_dim = 128
        self.tracker_dim = 128
        self.predict = True


class EncoderConfig:
    def __init__(self):
        self.max_vocab_size = 5000
        self.embedding_dim = 128
        self.hidden_mode = 'QRNN'
        self.num_hidden_layer = 1
        self.hidden_dim = 128
        self.window_sizes = [2, 2]

        self.dropout = 0.3

class MultiTaskConfig:

    def __init__(self):
        self.encoder_config = EncoderConfig()
        self.batch_size = 16
        self.data_root = '/Users/sunqf/startup/quotesbot/nlp-data/chinese_segment/data/'
        #self.data_root = '/home/sunqf/Work/chinese_segment/data'

        import os
        '''
        people2014 = TaggerConfig('people2014', [os.path.join(self.data_root, 'train/people2014.txt')], [], False)
        ctb = TaggerConfig('ctb8',
                           [os.path.join(self.data_root, 'train/ctb.train')],
                           [os.path.join(self.data_root, 'gold/ctb.gold')],
                           False)
        msr = TaggerConfig('msr',
                           [os.path.join(self.data_root, 'train/msr_training.utf8')],
                           [os.path.join(self.data_root, 'gold/msr_test_gold.utf8')],
                           False)

        pku = TaggerConfig('pku',
                           [os.path.join(self.data_root, 'train/pku_training.utf8')],
                           [os.path.join(self.data_root, 'gold/pku_test_gold.utf8')],
                           False)

        nlpcc = TaggerConfig('nlpcc',
                             [os.path.join(self.data_root, 'train/nlpcc2016-word-seg-train.dat'),
                                   os.path.join(self.data_root, 'train/nlpcc2016-wordseg-dev.dat')],
                             #[os.path.join(self.data_root, 'gold/nlpcc2016-wordseg-test.dat')],
                             [],
                             False)


        ctb_pos = TaggerConfig('ctb_pos',
                               [os.path.join(self.data_root, 'pos/ctb.pos.train')],
                               [os.path.join(self.data_root, 'pos/ctb.pos.gold')],
                               True)
        '''

        self.ctb_parser = ParserConfig('ctb_parser',
                                  [os.path.join(self.data_root, 'parser/ctb.parser.train')],
                                  [os.path.join(self.data_root, 'parser/ctb.parser.gold')])

        self.tasks = [self.ctb_parser]

        self.valid_size = 1000//self.batch_size

        # 字属性字典
        self.char_attr = './dict/word-type'
        # 词集合
        self.wordset = {}
        self.model_prefix = 'model/model'

        self.eval_step = 500

        self.epoches = 10


config = MultiTaskConfig()
encoder_config = config.encoder_config
parser_config = config.ctb_parser

parser_data = CTBParser(parser_config.train_paths, parser_config.test_paths)

vocab = Vocab.build(parser_data.word_counts, encoder_config.max_vocab_size)

char2attr = CharacterAttribute.load(config.char_attr)
gazetteers = [char2attr] + [Gazetteer.load(name, path) for name, path in config.wordset]

encoder = Encoder(len(vocab), gazetteers, encoder_config.embedding_dim,
                  encoder_config.hidden_mode, encoder_config.hidden_dim,
                  encoder_config.num_hidden_layer, encoder_config.window_sizes,
                  encoder_config.dropout)

task = ParserTask('ctb_parser', encoder, vocab, parser_config.hidden_dim, parser_config.tracker_dim)

train_data = list(parser_data.batch_train(vocab, gazetteers, config.batch_size))
test_data = list(parser_data.batch_test(vocab, gazetteers, config.batch_size))

valid_data = []
train_data, valid_data = train_test_split(train_data, test_size=1000//config.batch_size)

optimzer = torch.optim.Adam(task.params)

for epoch in range(10):

    total_loss = 0.
    total_correct = 0.
    total_count = 0

    seq_step = 50
    for batch_id, batch in enumerate(train_data, start=1):
        sentences, transitions = batch
        for begin in range(0, transitions.size(0), seq_step):
            task.train()
            task.zero_grad()

            loss, correct, count = task.range_loss(sentences, transitions, begin, begin+seq_step)
            total_loss += loss.data[0] * count
            total_correct += correct
            total_count += count

            loss.backward()
            optimzer.step()

        if batch_id % 200 == 0:
            valid_loss = 0.
            valid_correct = 0.
            valid_count = 0
            task.eval()
            for valid_batch in valid_data:
                sentences, transitions = batch
                loss, correct, count = task.loss(sentences, transitions)

                valid_loss += loss.data[0] * count
                valid_correct += correct
                valid_count += count

            print('train loss=%f\tacc=%0.6f\nvalid=%f\tacc=%0.6f' %
                  (total_loss / total_count, total_correct / total_count,
                   valid_loss / valid_count, valid_correct / valid_count))

            total_loss = 0.
            total_correct = 0.
            total_count = 0

            #sample
            import random
            sentences, transitions = valid_data[random.randint(0, len(valid_data)-1)]
            task.eval()
            print('\n'.join(task.sample(sentences, vocab)))


