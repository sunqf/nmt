

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from collections import defaultdict, namedtuple
import itertools
from .task import Task, Loader, TaskConfig
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence
from ..util.vocab import Vocab
from ..util.utils import replace_entity

import copy

import numpy as np


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

class BoundaryLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(BoundaryLSTM, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.dropout_layer = nn.Dropout(self.dropout)
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim, num_layers)

        self._begin = nn.Parameter(torch.FloatTensor(1, hidden_dim*2))

        self._end = nn.Parameter(torch.FloatTensor(1, hidden_dim*2))

    def begin(self):
        return self._begin

    def end(self):
        return self._end

    def forward_end(self, hidden):
        return self.forward([self._end]*len(hidden), hidden)

    def forward(self, input, hidden=None):
        """
        :param input: [tensor(1, input_dim)] * batch_size or list
        :param hidden: [tensor(1, hidden_dim*2)] * batch_size or None
        :return: [tensor(1, hidden_dim*2)] * batch_size,
        """
        if isinstance(input, list):
            input = torch.cat(input, 0)

        if hidden is None:
            hidden = [self._begin] * input.size(0)
        if isinstance(hidden, list):
            hidden = bundle(hidden)

        new_hx, new_cx = self.lstm_cell(self.dropout_layer(input), hidden)

        return unbundle([new_hx, new_cx])


# 依存树的组合函数
class Composition(nn.Module):
    def __init__(self, node_dim, relation_dim, dropout=0.2):
        super(Composition, self).__init__()

        self.node_dim = node_dim
        self.relation_dim = relation_dim
        self.dropout = dropout
        self.model = nn.Sequential(nn.Dropout(self.dropout),
                                   nn.Linear(node_dim * 2 + relation_dim, node_dim),
                                   nn.Tanh())

    def forward(self, heads, modifiers, relations):
        '''
        :param heads: [tensor(1, self.node_dim)] * batch_size  or  tensor(batch_size, self.node_dim)
        :param modifiers:  [tensor(1, self.node_dim)] * batch_size   or  tensor(batch_size, self.node_dim)
        :param relations: [tensor(1, self.relation_dim)] * batch_size   or  tensor(batch_size, self.relation_dim)
        :return: [tensor(1, self.node_dim)] * batch_size
        '''
        if isinstance(heads, list):
            heads = torch.cat(heads, 0)
        if isinstance(modifiers, list):
            modifiers = torch.cat(modifiers, 0)
        if isinstance(relations, list):
            relations = torch.cat(relations, 0)

        comp = self.model(torch.cat([heads, modifiers, relations], 1))

        return comp.split(1, 0)

class Node:
    def __init__(self, index, chars, pos=None, head_index=-1, relation=None, lefts=None, rights=None):
        self.index = index
        self.chars = chars if chars else []
        self.pos = pos
        self.head_index = head_index
        self.relation = relation
        self.lefts = lefts if lefts else []
        self.rights = rights if rights else []

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return '(%s\t%s\t%s)' % ('\t'.join([str(left) for relation, left in self.lefts]),
                                 ''.join(self.chars),
                                 '\t'.join([str(right) for relation, right in self.rights]))

    def __deepcopy__(self, memodict={}):

        index = copy.copy(self.index)
        chars = copy.copy(self.chars)
        pos = copy.copy(self.pos)
        head_index = copy.copy(self.head_index)
        relation = copy.copy(self.relation)
        lefts = copy.deepcopy(self.lefts, memodict)
        rights = copy.deepcopy(self.rights, memodict)
        new_node = Node(index, chars, pos, head_index, relation, lefts, rights)
        memodict[id(self)] = new_node
        return new_node

class Actions:
    SHIFT = 0
    APPEND = 1
    ARC_LEFT = 2
    ARC_RIGHT = 3
    #ROOT = 4
    max_len = 4

class Transition:
    def __init__(self, action, label):
        self.action = action
        self.label = label

    def __eq__(self, other):
        return hash(self) == hash(other) and self.action == other.action and self.label == other.label

    def __hash__(self):
        return hash(self.action) + hash(self.label)

    def __str__(self):
        return '%d,%s' % (self.action, self.label)

class UDTree:
    def __init__(self, nodes, root):
        self.nodes = nodes
        self.root = root

    def linearize(self):
        chars = list(itertools.chain.from_iterable([node.chars for node in self.nodes]))
        relations = [node.relation for node in self.nodes]
        pos = [node.pos for node in self.nodes]

        def dfs(node):
            for left in node.lefts:
                yield from dfs(left)

            yield Transition(Actions.SHIFT, node.pos)
            for c in node.chars[1:]:
                yield Transition(Actions.APPEND, None)

            for l in node.lefts[::-1]:
                yield Transition(Actions.ARC_LEFT, l.relation)

            for right in node.rights:
                yield from dfs(right)
                yield Transition(Actions.ARC_RIGHT, right.relation)


        return chars, (list(dfs(self.root)), relations, pos)

    def get_words(self):
        return [''.join(node.chars) for node in self.nodes]

    def to_line(self):
        return '\t'.join(['%s#%s#%d#%s' % (''.join(node.chars), node.pos, node.head_index, node.relation) for node in self.nodes])

    @staticmethod
    def create(chars, transitons):
        nodes = []
        stack = []
        char_index = 0
        word_index = 0
        for tran in transitons:
            if tran.action == Actions.SHIFT:
                new_node = Node(word_index, [chars[char_index]], tran.label)
                stack.append(new_node)
                nodes.append(new_node)
                char_index += 1
                word_index += 1
            elif tran.action == Actions.APPEND:
                stack[-1].chars.append(chars[char_index])
                char_index += 1
            elif tran.action == Actions.ARC_LEFT:
                head = stack.pop()
                modifier = stack.pop()
                modifier.head_index = head.index
                modifier.relation = tran.label
                head.lefts.append(modifier)
                stack.append(head)
            elif tran.action == Actions.ARC_RIGHT:
                modifier = stack.pop()
                head = stack[-1]
                head.rights.append(modifier)
                modifier.head_index = head.index
                modifier.relation = tran.label

        for node in nodes:
            node.lefts = list(reversed(node.lefts))

        stack[-1].relation = 'root'

        return UDTree(nodes, stack[-1])

    @staticmethod
    def parse_stanford_format(tokens):
        nodes = [Node(index, chars, pos, int(parent_id)-1, relation) for index, (chars, pos, parent_id, relation) in enumerate(tokens)]

        root_id = 0
        for id, (token, pos, parent_id, relation) in enumerate(tokens):
            parent_id = int(parent_id) - 1

            if parent_id == -1:
                root_id = id
            elif parent_id > id:
                nodes[parent_id].lefts.append(nodes[id])
            elif parent_id < id:
                nodes[parent_id].rights.append(nodes[id])

        def validate(tree):
            gold = tree.to_line()

            chars, (trans, _, _) = tree.linearize()
            new_tree = UDTree.create(chars, trans)
            new_line = new_tree.to_line()

            return gold == new_line

        tree = UDTree(nodes, nodes[root_id])

        if validate(tree) is False:
            print('invalid tree:')
            print(tree.to_line())
            return None

        return UDTree(nodes, nodes[root_id])


class TransitionClassifier(nn.Module):

    def __init__(self, input_dim, num_transition, dropout=0.2):
        super(TransitionClassifier, self).__init__()

        self.num_transition = num_transition
        self.input_dim = input_dim

        self.dropout = dropout

        self.ffn = nn.Sequential(nn.Dropout(dropout),
                                 nn.Linear(self.input_dim, self.input_dim//2),
                                 nn.Sigmoid(),
                                 nn.Dropout(dropout),
                                 nn.Linear(self.input_dim//2, self.input_dim//2),
                                 nn.Sigmoid(),
                                 nn.Linear(self.input_dim//2, num_transition)
                                 )

    def forward(self, buffer_hiddens, stack_hiddens, transition_hiddens, mask):

        buffers = bundle([b[-1] for b in buffer_hiddens])[0]
        stacks1 = bundle([s[-1] for s in stack_hiddens])[0]
        stacks2 = bundle([s[-2] for s in stack_hiddens])[0]
        transitions = bundle([t[-1] for t in transition_hiddens])[0]
        features = torch.cat([buffers, stacks1, stacks2, transitions], 1)

        return F.log_softmax(self.ffn(features) * mask, 1)


class State:
    def __init__(self, nodes, buffer, buffer_hidden, stack, stack_hidden, transitions, transition_hidden, score=0.0):
        self.nodes = nodes
        self.buffer = buffer
        self.buffer_hidden = buffer_hidden
        self.stack = stack
        self.stack_hidden = stack_hidden
        self.transitions = transitions
        self.transitions_hidden = transition_hidden
        self.score = score

    def __deepcopy__(self, memodict={}):
        nodes = copy.deepcopy(self.nodes, memodict)
        buffer = copy.copy(self.buffer)
        buffer_hidden = copy.copy(self.buffer_hidden)
        stack = copy.copy(self.stack)
        stack_hidden = copy.copy(self.stack_hidden)
        transitions = copy.copy(self.transitions)
        transitions_hidden = copy.copy(self.transitions_hidden)
        score = copy.copy(self.score)

        new_state = State(nodes, buffer, buffer_hidden, stack, stack_hidden, transitions, transitions_hidden, score)
        memodict[id(self)] = new_state
        return new_state


# reference https://www.researchgate.net/profile/Yue_Zhang4/publication/266376262_Character-Level_Chinese_Dependency_Parsing/links/542e18030cf277d58e8e9908/Character-Level-Chinese-Dependency-Parsing.pdf
class ArcStandard(nn.Module):
    def __init__(self, input_dim, hidden_dim, transition_dict, relation_dict, pos_dict, dropout=0.2):
        super(ArcStandard, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.transition_dict = transition_dict
        self.relation_dict = relation_dict
        self.pos_dict = pos_dict
        self.dropout = dropout

        # buffer hidden
        self.buffer_dim = self.hidden_dim
        self.buffer_lstm = BoundaryLSTM(input_dim, self.buffer_dim, 1, self.dropout)

        #self.buffer_lstm = nn.LSTM(input_dim, self.buffer_dim, 1, dropout=self.dropout, bidirectional=False)

        # word embedding from char rnn
        self.word_dim = self.hidden_dim
        self.pos_dim = self.hidden_dim // 2
        self.pos_emb = nn.Embedding(len(self.pos_dict), self.pos_dim)
        self.word_lstm = BoundaryLSTM(input_dim+self.pos_dim, self.word_dim, 1, self.dropout)

        # stack hidden
        self.stack_dim = self.hidden_dim
        self.stack_lstm = BoundaryLSTM(self.word_dim, self.stack_dim, 1, self.dropout)

        self.relation_dim = self.word_dim // 2
        self.relation_emb = nn.Embedding(len(self.relation_dict), self.relation_dim)
        # compose head and modifier
        self.dependency_compsition = Composition(self.word_dim, self.relation_dim)

        # transition id to embedding
        self.transition_dim = self.hidden_dim // 4
        self.transition_emb = nn.Embedding(len(self.transition_dict), self.transition_dim)
        # transition hidden
        self.transition_lstm = BoundaryLSTM(self.transition_dim, self.transition_dim, 1, self.dropout)


        #self.encoder = QRNN(input_dim, feature_dim, 1,
        #                    window_sizes=3, dropout=dropout)
        self.transition_classifier = TransitionClassifier(self.buffer_dim + self.stack_dim * 2 + self.transition_dim,
                                                          len(self.transition_dict), self.dropout)


    def _buffer(self, sentences):
        if isinstance(sentences, PackedSequence):
            buffers, lengths = pad_packed_sequence(sentences, batch_first=False)
        else:
            raise 'buffers must be PackedSequence'

        buffers = [list(torch.split(b.squeeze(1)[:length], 1, 0))
                   for b, length in zip(torch.split(buffers, 1, 1), lengths)]

        buffers = [list(reversed(b)) for b in buffers]

        buffer_hiddens = [[self.buffer_lstm.begin()] for b in buffers]
        max_length = lengths[0]
        for t in range(max_length):
            indexes = [i for i, _ in enumerate(buffers) if len(buffers[i]) > t]
            inputs = [buffers[i][t] for i in indexes]
            hidden = [buffer_hiddens[i][-1] for i in indexes]
            new_hiddens = self.buffer_lstm(inputs, hidden)

            for i, new in zip(indexes, new_hiddens):
                buffer_hiddens[i].append(new)

        return buffers, buffer_hiddens, lengths


    def _update_state(self, state_t, transition_id_t, scores):
        """
        :param state_t: [state] * batch_size
        :param transition_id_t: Variable(LongTensor(batch_size))
        :param scores: [float] * batch_size
        :return:
        """
        def word_emb(chars):
            return chars[-1][:, 0:self.word_dim]

        transition_t = self.transition_dict.get_word(transition_id_t.data)
        # update char rnn
        update_nodes, update_char_inputs, update_pos, update_char_hiddens = [], [], [], []
        for state, t in zip(state_t, transition_t):
            if t.action == Actions.SHIFT:
                next_char = state.buffer.pop()
                state.buffer_hidden.pop()
                node = Node(len(state.nodes), [self.word_lstm.begin()], t.label)
                state.nodes.append(node)

                update_nodes.append(node)
                update_char_inputs.append(next_char)
                update_pos.append(node.pos)
                update_char_hiddens.append(node.chars[-1])
            elif t.action == Actions.APPEND:
                next_char = state.buffer.pop()
                state.buffer_hidden.pop()
                node = state.nodes[-1]

                update_nodes.append(node)
                update_char_inputs.append(next_char)
                update_pos.append(node.pos)
                update_char_hiddens.append(node.chars[-1])

        if len(update_nodes) > 0:
            update_pos_embs = self.pos_emb(Variable(torch.LongTensor(self.pos_dict.convert(update_pos)), requires_grad=False))
            update_char_inputs = torch.cat(update_char_inputs, 0)
            new_char_hidden = self.word_lstm(torch.cat([update_char_inputs, update_pos_embs], 1), update_char_hiddens)
            for node, new in zip(update_nodes, new_char_hidden):
                node.chars.append(new)

        need_comp_states, need_comp_heads, need_comp_modifiers, need_comp_relations = [], [], [], []
        for state, t, score in zip(state_t, transition_t, scores):
            state.score = score
            if t.action == Actions.SHIFT:
                state.stack.append(word_emb(state.nodes[-1].chars))
            elif t.action == Actions.APPEND:
                state.stack.pop()
                state.stack_hidden.pop()
                state.stack.append(word_emb(state.nodes[-1].chars))
            elif t.action == Actions.ARC_LEFT:
                head = state.stack.pop()
                modifier = state.stack.pop()
                head_hidden = state.stack_hidden.pop()
                modifier_hidden = state.stack_hidden.pop()
                need_comp_states.append(state)
                need_comp_heads.append(head)
                need_comp_modifiers.append(modifier)
                need_comp_relations.append(t.label)
            elif t.action == Actions.ARC_RIGHT:
                modifier = state.stack.pop()
                head = state.stack.pop()
                modifier_hidden = state.stack_hidden.pop()
                head_hidden = state.stack_hidden.pop()
                need_comp_states.append(state)
                need_comp_heads.append(head)
                need_comp_modifiers.append(modifier)
                need_comp_relations.append(t.label)

        # update composition node
        if len(need_comp_states) > 0:
            relation_emb = self.relation_emb(Variable(torch.LongTensor(self.relation_dict.convert(need_comp_relations)), requires_grad=False))
            new_heads = self.dependency_compsition(need_comp_heads, need_comp_modifiers, relation_emb)
            for state, new in zip(need_comp_states, new_heads):
                state.stack.append(new)

        # update stack hidden
        new_stack_hiddens = self.stack_lstm([state.stack[-1] for state in state_t],
                                            [state.stack_hidden[-1] for state in state_t])

        for state, stack_hidden in zip(state_t, new_stack_hiddens):
            state.stack_hidden.append(stack_hidden)

        # update transition hidden
        transition_embs = self.transition_emb(transition_id_t)
        new_transition_hiddens = self.transition_lstm(transition_embs,
                                                      [state.transitions_hidden[-1] for state in state_t])
        for state, transition, transition_hidden in zip(state_t, transition_t, new_transition_hiddens):
            state.transitions.append(transition)
            state.transitions_hidden.append(transition_hidden)

        return state_t

    def loss(self, sentences, gold):
        '''
        :param sentences: [length, batch, dim]
        :param gold_transitions: [transition] * batch_size
        :return:
        '''
        gold_transitions, gold_relations, gold_pos = gold

        #sentences, _ = self.encoder(sentences)

        buffers, buffer_hiddens, lengths = self._buffer(sentences)

        batch_size = len(lengths)
        # [batch_size * [stack_size * [node]]]
        stacks = [[]] * batch_size

        states = [State([], buffer, buffer_hidden,
                        [], [self.stack_lstm.begin(), self.stack_lstm.begin()],
                        [], [self.transition_lstm.begin()])
                  for buffer, buffer_hidden, stack, pos in zip(buffers, buffer_hiddens, stacks, gold_pos)]

        max_transition_length = gold_transitions.size(0)


        transition_losses = []
        transition_correct = 0
        transition_count = 1e-5

        pos_loss = [0]
        pos_correct = 0
        pos_count = 1e-5

        seg_loss = []
        seg_correct = 0
        seg_count = 1e-5

        for t in range(max_transition_length):
            state_t, transition_id_t, mask_t = [], [], []
            for state, transition_id in zip(states, gold_transitions[t].data):
                if transition_id >= 0:
                    state_t.append(state)
                    transition_id_t.append(transition_id)
                    transition = self.transition_dict.get_word(transition_id)
                    #assert self.check(state.buffer, state.stack, state.transitions, transition)
                    mask_t.append(self.mask(len(state.buffer), len(state.stack), state.transitions[-1] if len(state.transitions) > 0 else None))
            if len(transition_id_t) == 0:
                break

            transition_mask = torch.cat(mask_t, 0)
            transition_log_prob = self.transition_classifier([state.buffer_hidden for state in state_t],
                                                         [state.stack_hidden for state in state_t],
                                                         [state.transitions_hidden for state in state_t],
                                                         transition_mask)

            # caculate loss
            transition_id_t = Variable(torch.LongTensor(transition_id_t), requires_grad=False)

            transition_losses.append(F.nll_loss(transition_log_prob,
                                            transition_id_t,
                                            size_average=False))

            transition_log_max, transition_argmax = transition_log_prob.max(1)
            transition_correct += (transition_argmax.data == transition_id_t.data).sum()
            transition_count += transition_id_t.nelement()

            scores = torch.FloatTensor([state.score for state in state_t]) + transition_log_max.data
            self._update_state(state_t, transition_id_t, scores)

        return (F.reduce(lambda x, y: x + y, transition_losses), transition_correct, transition_count), (sum(pos_loss)/pos_count, pos_correct, pos_count)

    def mask(self, buffer_len, stack_len, last_transition):
        action_blacklist = set()

        if buffer_len == 0:
            action_blacklist.add(Actions.SHIFT)
            action_blacklist.add(Actions.APPEND)

        if stack_len == 0:
            action_blacklist.add(Actions.APPEND)

        if last_transition is None:
            action_blacklist.update([Actions.APPEND, Actions.ARC_LEFT, Actions.ARC_RIGHT])
        elif last_transition.action == Actions.ARC_LEFT or last_transition.action == Actions.ARC_RIGHT:
            action_blacklist.add(Actions.APPEND)

        if stack_len < 2:
            action_blacklist.add(Actions.ARC_LEFT)
            action_blacklist.add(Actions.ARC_RIGHT)
        '''
        transition_whitelist = set()
        if last_transition is not None and last_transition.action == Actions.SHIFT:
            action_blacklist.add(Actions.APPEND)
            transition_whitelist.add(Transition(Actions.APPEND, last_transition.label))

        def to_mask(t):
            if t.action in action_blacklist and t is not transition_whitelist:
                return 0
            return 1
        '''

        def to_mask(t):
            if t.action in action_blacklist:
                return 0
            return 1

        masked = Variable(torch.FloatTensor([[to_mask(t) for t in self.transition_dict.words]]), requires_grad=False)
        return masked

    @staticmethod
    def check(buffer, stack, prev_transitions, curr_transition):

        if len(prev_transitions) == 0:
            if curr_transition.action != Actions.SHIFT:
                return False
        else:
            '''
            if prev_transitions[-1].action in [Actions.SHIFT, Actions.APPEND] \
                    and curr_transition.action == Actions.APPEND \
                    and prev_transitions[-1].label != curr_transition.label:
                return False
            '''

            prev_action = prev_transitions[-1].action
            curr_action = curr_transition.action

            if curr_action == Actions.SHIFT:
                if len(buffer) == 0:
                    return False
            elif curr_action == Actions.APPEND:
                if prev_action != Actions.SHIFT and prev_action != Actions.APPEND:
                    return False
                if len(buffer) == 0 or len(stack) == 0:
                    return False

            elif curr_action == Actions.ARC_LEFT or curr_action == Actions.ARC_RIGHT:
                if len(stack) < 2:
                    return False
        return True

    def parse(self, sentences, beam_size=10):

        #sentences, _ = self.encoder(sentences)

        buffers, buffer_hiddens, lengths = self._buffer(sentences)

        return [self.parse_one(buffer[0:length], buffer_hidden, beam_size)
                for buffer, buffer_hidden, length in zip(buffers, buffer_hiddens, lengths)]

    def parse_one(self, sentence, buffer_hidden, beam_size=10):

        length = len(sentence) * 2 - 1

        topK = []
        topK.append(State([], sentence, buffer_hidden,
                          [], [self.stack_lstm.begin(), self.stack_lstm.begin()],
                          [], [self.transition_lstm.begin()]))

        next1 = []

        for step in range(length):
            transition_mask = torch.cat([self.mask(len(state.buffer),
                                                   len(state.stack),
                                                   state.transitions[-1] if len(state.transitions) else None) for state in topK],
                                        0)
            transition_log_prob = self.transition_classifier([state.buffer_hidden for state in topK],
                                                         [state.stack_hidden for state in topK],
                                                         [state.transitions_hidden for state in topK],
                                                         transition_mask)

            transition_log_prob = transition_log_prob.split(1, 0)



            step1, step2 = [], []
            for state, log_probs in zip(topK, transition_log_prob):
                for transition_id, log_prob in enumerate(log_probs.squeeze().data):
                    transition = self.transition_dict.get_word(transition_id)
                    if self.check(state.buffer, state.stack, state.transitions, transition):
                        if transition.action in [Actions.SHIFT, Actions.ARC_LEFT, Actions.ARC_RIGHT]:
                            step1.append((state, transition, state.score+log_prob))
                        else:
                            # append = shift + reduce
                            step2.append((state, transition, state.score+log_prob))

            sorted_cands = sorted(step1 + next1, key=lambda c: c[2], reverse=True)

            topK = sorted_cands[0: beam_size]

            topK = self._update_state([copy.deepcopy(state) for state, _, _ in topK],
                                      Variable(torch.LongTensor(self.transition_dict.convert([transition for _, transition, _ in topK]))),
                                      torch.FloatTensor([score for _, _, score in topK]))
            next1 = step2

        return topK[0].transitions, topK[0].score

class ParserConfig(TaskConfig):

    def __init__(self, name, train_paths, test_paths):
        self.name = name
        self.train_paths = train_paths
        self.test_paths = test_paths

        self.hidden_dim = 128
        self.dropout = 0.2
        self.shared_weight_decay = 1e-6
        self.task_weight_decay = 1e-6

        self._loader = None

    def loader(self):
        if self._loader is None:
            self._loader = CTBParseData(self.train_paths, self.test_paths)

        return self._loader

    def create_task(self, shared_vocab, shared_encoder):

        loader = self.loader()

        return ParserTask(self.name,
                          shared_encoder, shared_vocab,
                          loader.transition_dict, loader.relation_dict, loader.pos_dict,
                          self.hidden_dim,
                          self.dropout,
                          self.shared_weight_decay,
                          self.task_weight_decay)


class ParserTask(Task):
    def __init__(self, name, encoder, vocab,
                 transition_dict, relation_dict, pos_dict,
                 hidden_dim,
                 dropout,
                 shared_weight_decay,
                 task_weight_decay):
        super(ParserTask, self).__init__()

        self.name = name
        self.vocab = vocab
        self.transition_dict = transition_dict
        self.relation_dict = relation_dict
        self.pos_dict = pos_dict
        self.shared_encoder = encoder

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.shared_weight_decay = shared_weight_decay
        self.task_weight_decay = task_weight_decay

        self.parser = ArcStandard(self.shared_encoder.output_dim(),
                                  self.hidden_dim,
                                  self.transition_dict,
                                  self.relation_dict,
                                  self.pos_dict,
                                  self.dropout)


        self.params = [{'params': self.shared_encoder.parameters(), 'weight_decay': self.shared_weight_decay},
                       #{'params': self.task_encoder.parameters(), 'weight_decay': task_weight_decay},
                       {'params': self.parser.parameters(), 'weight_decay': self.task_weight_decay}]


    def forward(self, sentences, transitions):
        sentences, gazetteers = sentences

        feature = self.shared_encoder(sentences, gazetteers)
        return self.parser.loss(feature, transitions)

    def _to_cuda(self, batch_data):

        (sentences, gazes), (transitions, relations, pos) = batch_data

        return ((PackedSequence(sentences.data.cuda(), sentences.batch_sizes),
                 PackedSequence(gazes.data.cuda(), gazes.batch_sizes)),
                (transitions.cuda(), relations.cuda(), pos.cuda())
                )

    def loss(self, batch_data, use_cuda=False):

        if use_cuda:
            batch_data = self._to_cuda(batch_data)

        sentences, reference = batch_data
        (loss, acc, count), _ = self.forward(sentences, reference)

        return loss/count

    def parse(self, sentences, beam_size=5):
        sentences, gazetteers = sentences

        features = self.shared_encoder(sentences, gazetteers)

        return self.parser.parse(features, beam_size)

    def sample(self, batch_data, use_cuda=False):

        if use_cuda:
            batch_data = self._to_cuda(batch_data)

        sentences, (gold_trans, _, _) = batch_data
        pred_transitions_and_score = self.parse(sentences, 5)

        sentences, lengths = pad_packed_sequence(sentences[0], batch_first=True, padding_value=-1)
        sentences = [self.vocab.get_word(sentence[0, :length].data.numpy())
                     for sentence, length in zip(sentences.split(1, 0), lengths)]

        gold_trans = [gold_trans[:, sen_id] for sen_id in range(len(sentences))]
        gold_trans = [t.masked_select(t >= 0) for t in gold_trans]

        samples = [(score, UDTree.create(sentence, self.transition_dict.get_word(gold.data)).to_line(),
                    UDTree.create(sentence, pred).to_line())
                   for sentence, gold, (pred, score) in zip(sentences, gold_trans, pred_transitions_and_score)]

        return ['score=%f\nref:  %s\npred: %s\n' % (score, gold, pred) for score, gold, pred in samples if gold != pred]


    def evaluation(self, data, use_cuda=False):

        seg_correct = 0
        pos_correct = 0

        uas_correct = 0
        las_correct = 0

        gold_count = 0
        pred_count = 0

        labeled_arc_correct = 0
        unlabeled_arc_correct = 0


        for batch in data:
            if use_cuda:
                batch = self._to_cuda(batch)

            sentences, (transitions, _, _) = batch
            pred_trans = self.parse(sentences)

            sentences, lengths = pad_packed_sequence(sentences[0], batch_first=True, padding_value=-1)
            sentences = [sentence[0, :length] for sentence, length in zip(sentences.split(1, 0), lengths)]
            for sen_id, (sentence, (sen_pred_trans, score)) in enumerate(zip(sentences, pred_trans)):
                sen_gold_trans = transitions[:, sen_id]
                sen_gold_trans = sen_gold_trans.masked_select(sen_gold_trans >= 0)

                gold_tree = UDTree.create(self.vocab.get_word(sentence.data.numpy()), self.transition_dict.get_word(sen_gold_trans.data))
                pred_tree = UDTree.create(self.vocab.get_word(sentence.data.numpy()), sen_pred_trans)

                gold_nodes = gold_tree.nodes
                pred_nodes = pred_tree.nodes

                gold_count += len(gold_nodes)
                pred_count += len(pred_nodes)

                gold_char_index, gold_word_index = 0, 0
                pred_char_index, pred_word_index = 0, 0

                matched_nodes = []

                while gold_word_index < len(gold_nodes) and pred_word_index < len(pred_nodes):

                    if gold_nodes[gold_word_index].chars == pred_nodes[pred_word_index].chars:
                        seg_correct += 1
                        matched_nodes.append((gold_word_index, pred_word_index))
                        if gold_nodes[gold_word_index].pos == pred_nodes[pred_word_index].pos:
                            pos_correct += 1

                    gold_char_index += len(gold_nodes[gold_word_index])
                    pred_char_index += len(pred_nodes[pred_word_index])

                    gold_word_index += 1
                    pred_word_index += 1

                    while gold_char_index != pred_char_index:
                        if gold_char_index < pred_char_index:
                            gold_char_index += len(gold_nodes[gold_word_index])
                            gold_word_index += 1
                        elif gold_char_index > pred_char_index:
                            pred_char_index += len(pred_nodes[pred_word_index])
                            pred_word_index += 1


                pred2gold = {pred: gold for gold, pred in matched_nodes}

                for gold_modifier, pred_modifier in matched_nodes:
                    pred_relation = pred_nodes[pred_modifier].relation
                    pred_head = pred_nodes[pred_modifier].head_index

                    gold_head = gold_nodes[gold_modifier].head_index
                    gold_relation = gold_nodes[gold_modifier].relation

                    if (pred_head == gold_head == -1) or pred2gold.get(pred_head, -100) == gold_head:
                        unlabeled_arc_correct += 1
                        if gold_relation == pred_relation:
                            labeled_arc_correct += 1

        safe_div = lambda a, b : a / (b + 1e-10)
        seg_prec = safe_div(seg_correct, pred_count)
        seg_recall = safe_div(seg_correct, gold_count)
        seg_f = safe_div(2*seg_prec*seg_recall, seg_prec+seg_recall)

        pos_prec = safe_div(pos_correct, pred_count)
        pos_recall = safe_div(pos_correct, gold_count)
        pos_f = safe_div(2*pos_prec*pos_recall, pos_prec+pos_recall)

        unlabeled_prec = safe_div(unlabeled_arc_correct, pred_count)
        unlabeled_recall = safe_div(unlabeled_arc_correct, gold_count)
        unlabeled_f = safe_div(2*unlabeled_prec*unlabeled_recall, unlabeled_prec+unlabeled_recall)

        labeled_prec = safe_div(labeled_arc_correct, pred_count)
        labeled_recall = safe_div(labeled_arc_correct, gold_count)
        labeled_f = safe_div(2*labeled_prec*labeled_recall, labeled_prec+labeled_recall)


        return {'seg prec':seg_prec, 'seg recall':seg_recall, 'seg F-score':seg_f,
                'pos pred':pos_prec, 'pos recall':pos_recall, 'pos F-score':pos_f,
                'ua prec':unlabeled_prec, 'ua recall':unlabeled_recall, 'ua F-score':unlabeled_f,
                'la prec':labeled_prec, 'la recall': labeled_recall, 'la F-score':labeled_f
                }


class CTBParseData(Loader):
    def __init__(self, train_paths, test_paths):
        self.train_data = self.load(train_paths)

        self.word_counts = defaultdict(int)
        self.pos_counts = defaultdict(int)
        self.transition_counts = defaultdict(int)
        self.relation_counts = defaultdict(int)
        for chars, (transitions, relations, pos) in self.train_data:
            for t in transitions:
                self.transition_counts[t] += 1
                self.relation_counts[t.label] += 1
            for p in pos:
                self.pos_counts[p] += 1

            for char in chars:
                self.word_counts[char] += 1
        self.test_data = self.load(test_paths)

        self.train_data = sorted(self.train_data, key=lambda item: len(item[0]), reverse=True)
        self.test_data = sorted(self.test_data, key=lambda item: len(item[0]), reverse=True)

        self.min_count = 5
        transitions = [k if k.label is None or self.relation_counts[k.label] > self.min_count else Transition(k.action, 'UNK_RELATION')
                                      for k, v in self.transition_counts.items()]

        self.transition_dict = Vocab(transitions, unk=None)
        self.pos_dict = Vocab([k for k, v in self.pos_counts.items() if v > self.min_count], unk='UNK_POS')

        relations = set([t.label for t in transitions if t.action in [Actions.ARC_LEFT, Actions.ARC_RIGHT]])
        self.relation_dict = Vocab(relations, unk=None)

    def load(self, paths):
        data = []
        bad_data_count = 0
        for path in paths:
            with open(path) as file:
                sentences = file.read().strip().split('\n\n')
                # uniq operation
                #sentences = set(sentences)
                for sentence in sentences:
                    words = sentence.strip().split('\n')
                    if len(words) > 0:
                        words = [word.split('\t') for word in words]
                        words = [([char if type == '@zh_char@' else type for type, char in replace_entity(word)], pos, parent_id, relation)
                                 for _, word, _, pos, _, _, parent_id, relation, _, _ in words]
                        tree = UDTree.parse_stanford_format(words)

                        if tree is None:
                            bad_data_count += 1
                        else:
                            data.append(tree.linearize())

        print(bad_data_count)
        return data

    def _batch(self, data, vocab, gazetteers, batch_size):

        gazetteers_dim = sum([c.length() for c in gazetteers])

        for begin in range(0, len(data), batch_size):
            batch = data[begin:begin+batch_size]

            batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)
            sen_lens = [len(s) for s, _ in batch]
            max_sen_len = max(sen_lens)
            max_tran_len = max([len(trans) for _, (trans, _, _) in batch])
            max_pos_len = max([len(pos) for _, (_, _, pos) in batch])
            sentences = torch.LongTensor(max_sen_len, len(batch)).fill_(0)
            gazes = torch.FloatTensor(max_sen_len, len(batch), gazetteers_dim).fill_(0)
            transitions = torch.LongTensor(max_tran_len, len(batch)).fill_(-1)
            relations = torch.LongTensor(max_pos_len, len(batch)).fill_(-1)
            pos = torch.LongTensor(max_pos_len, len(batch)).fill_(-1)
            for id, (words, (trans, rel, str_pos)) in enumerate(batch):
                sen_len = len(words)
                sentences[:sen_len, id] = torch.LongTensor(vocab.convert(words))
                gazes[0:sen_len, id] = torch.cat([torch.FloatTensor(gazetteer.convert(words)) for gazetteer in gazetteers], -1)
                tran_len = len(trans)
                transitions[:tran_len, id] = torch.LongTensor(self.transition_dict.convert(trans))
                relations[:len(rel), id] = torch.LongTensor(self.relation_dict.convert(rel))
                pos[:len(str_pos), id] = torch.LongTensor(self.pos_dict.convert(str_pos))
            yield ((pack_padded_sequence(Variable(sentences), sen_lens),
                    pack_padded_sequence(Variable(gazes), sen_lens)),
                   (Variable(transitions), Variable(relations), Variable(pos)))

    def batch_train(self, vocab, gazetteers, batch_size):
        return self._batch(self.train_data, vocab, gazetteers, batch_size)

    def batch_test(self, vocab, gazetteers, batch_size):
        return self._batch(self.test_data, vocab, gazetteers, batch_size)
