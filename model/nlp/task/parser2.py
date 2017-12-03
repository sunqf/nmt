import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from collections import defaultdict, namedtuple, deque
import itertools
from ..layer.encoder import Encoder
from ..layer.qrnn import QRNN
from ..layer.crf import CRFLayer
from .task import Task, Loader, TaskConfig
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence
from ..util.vocab import Vocab, CharacterAttribute, Gazetteer
from ..util.utils import replace_entity
from sklearn.model_selection import train_test_split

import copy

import numpy as np

# word-level action reference https://www.researchgate.net/profile/Yue_Zhang4/publication/266376262_Character-Level_Chinese_Dependency_Parsing/links/542e18030cf277d58e8e9908/Character-Level-Chinese-Dependency-Parsing.pdf
# character-level action reference http://www.aclweb.org/anthology/P12-1110

class Node:
    def __init__(self, index, chars, pos=None, parent_index=-1, relation=None, lefts=None, rights=None):
        self.index = index
        self.chars = chars
        self.pos = pos
        self.parent_index = parent_index
        self.relation = relation
        self.lefts = lefts if lefts else []
        self.rights = rights if rights else []

    def __str__(self):
        return '(%s\t%s\t%s)' % ('\t'.join([str(left) for relation, left in self.lefts]),
                                 ''.join(self.chars),
                                 '\t'.join([str(right) for relation, right in self.rights]))

    def __deepcopy__(self, memodict={}):

        index = copy.copy(self.index)
        chars = copy.copy(self.chars)
        pos = copy.copy(self.pos)
        parent_index = copy.copy(self.parent_index)
        relation = copy.copy(self.relation)
        lefts = copy.deepcopy(self.lefts, memodict)
        rights = copy.deepcopy(self.rights, memodict)
        new_node = Node(index, chars, pos, parent_index, relation, lefts, rights)
        memodict[id(self)] = new_node
        return new_node

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

            yield Transitions.SHIFT
            for c in node.chars[1:]:
                yield Transitions.APPEND

            for l in range(len(node.lefts)):
                yield Transitions.ARC_LEFT

            for right in node.rights:
                yield from dfs(right)

            for r in range(len(node.rights)):
                yield Transitions.ARC_RIGHT

        return chars, (list(dfs(self.root)), relations, pos)

    def get_words(self):
        return [''.join(node.chars) for node in self.nodes]

    def to_line(self):
        return '\t'.join(['%s#%d' % (''.join(node.chars), node.parent_index) for node in self.nodes])

    @staticmethod
    def create(chars, transitons):

        nodes = []
        stack = []
        char_index = 0
        word_index = 0
        for tran in transitons:
            if tran == Transitions.SHIFT:
                new_node = Node(word_index, [chars[char_index]])
                stack.append(new_node)
                nodes.append(new_node)
                char_index += 1
                word_index += 1
            elif tran == Transitions.APPEND:
                stack[-1].chars.append(chars[char_index])
                char_index += 1
            elif tran == Transitions.ARC_LEFT:
                first = stack.pop()
                second = stack.pop()
                second.parent_index = first.index
                first.lefts.append(second)
                stack.append(first)
            elif tran == Transitions.ARC_RIGHT:
                first = stack.pop()
                second = stack[-1]
                second.rights.append(first)
                first.parent_index = second.index

        return UDTree(nodes, stack[-1])

    @staticmethod
    def parse_stanford_format(tokens):
        nodes = [Node(index, chars, pos) for index, (chars, pos, _, _) in enumerate(tokens)]

        root_id = 0
        for id, (token, pos, parent_id, relation) in enumerate(tokens):
            parent_id = int(parent_id)
            if parent_id == 0:
                root_id = id
            elif parent_id > id:
                nodes[parent_id-1].lefts.append(nodes[id])
            elif parent_id < id:
                nodes[parent_id-1].rights.append(nodes[id])

        return UDTree(nodes, nodes[root_id])




class Transitions:
    SHIFT = 0
    APPEND = 1
    ARC_LEFT = 2
    ARC_RIGHT = 3
    #ROOT = 4
    max_len = 4

class TransitionClassifier(nn.Module):

    def __init__(self, input_dim, num_action):
        super(TransitionClassifier, self).__init__()

        self.num_action = num_action
        self.input_dim = input_dim

        self.feature_dim = input_dim * 6 + 1
        self.ffn = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim//2),
                                 nn.Sigmoid(),
                                 nn.Linear(self.feature_dim//2, self.feature_dim//4),
                                 nn.Sigmoid(),
                                 nn.Linear(self.feature_dim//4, num_action)
                                 )

        self.stack_null = nn.Parameter(torch.FloatTensor(1, input_dim).uniform_(-1/input_dim, 1/input_dim))
        self.buffer_null = nn.Parameter(torch.FloatTensor(1, input_dim).uniform_(-1/input_dim, 1/input_dim))

    def forward(self, buffers, stacks):

        features = torch.cat([self.feature(buffer, stack) for buffer, stack in zip(buffers, stacks)], 0)

        return self.ffn(features)

    def word_feature(self, chars):
        return torch.cat([chars[-1][:, 0:self.input_dim//2], chars[0][:, self.input_dim//2:]], -1)

    def phrase_feature(self, node):
        return

    def feature(self, buffer, stack):

        b0 = buffer[-1] if len(buffer) > 0 else self.buffer_null
        b1 = buffer[-2] if len(buffer) > 1 else self.buffer_null
        b2 = buffer[-3] if len(buffer) > 2 else self.buffer_null

        s0 = self.word_feature(stack[-1].chars) if len(stack) > 0 else self.stack_null
        s1 = self.word_feature(stack[-2].chars) if len(stack) > 1 else self.stack_null
        s2 = self.word_feature(stack[-3].chars) if len(stack) > 2 else self.stack_null

        word_length = len(stack[-1].chars) if len(stack) > 0 else 0
        word_length = Variable(b0.data.new([[word_length]]), requires_grad=False)

        return torch.cat([b0, b1, b2, s0, s1, s2, word_length], -1)

class RelationClassifier(nn.Module):
    def __init__(self, input_dim, num_relation):
        super(RelationClassifier, self).__init__()

        self.input_dim = input_dim
        self.num_relation = num_relation

        self.ffn = nn.Sequential(nn.Linear(input_dim * 2, input_dim),
                                 nn.Sigmoid(),
                                 nn.Linear(input_dim, num_relation))

        self.stack_null = nn.Parameter(torch.FloatTensor(1, input_dim).uniform_(-1/input_dim, 1/input_dim))
        self.buffer_null = nn.Parameter(torch.FloatTensor(1, input_dim).uniform_(-1/input_dim, 1/input_dim))

        def forward(self, children, heads):
            features = torch([self.feature(child, head) for child, head in zip(children, heads)], 0)
            return self.ffn(features)

        def word_feature(self, chars):
            return torch.cat([chars[-1][:, 0:self.input_dim // 2], chars[0][:, self.input_dim // 2:]], -1)

        def feature(self, child, head):
            child_feature = self.word_feature(child.chars)
            head_feature = self.word_feature(head.chars)

            return torch.cat([child_feature, head_feature], -1)


class POSClassifier(nn.Module):
    def __init__(self, input_dim, num_pos):
        super(POSClassifier, self).__init__()

        self.input_dim = input_dim
        self.num_pos = num_pos

        self.ffn = nn.Sequential(nn.Linear(input_dim * 2, input_dim),
                                 nn.Sigmoid(),
                                 nn.Linear(input_dim, num_pos))

        self.stack_null = nn.Parameter(torch.FloatTensor(1, input_dim).uniform_(-1/input_dim, 1/input_dim))
        self.buffer_null = nn.Parameter(torch.FloatTensor(1, input_dim).uniform_(-1/input_dim, 1/input_dim))

    def forward(self, buffers, stacks):

        features = torch([self.feature(buffer, stack)  for buffer, stack in zip(buffers, stacks)], 0)
        return self.ffn(features)

    def word_feature(self, chars):
        return torch.cat([chars[-1][:, 0:self.input_dim//2], chars[0][:, self.input_dim//2:]], -1)

    def feature(self, buffer, stack):

        b0 = buffer[-1] if len(buffer) > 0 else self.buffer_null

        s0 = self.word_feature(stack[-1].chars) if len(stack) > 0 else self.stack_null
        s1 = self.word_feature(stack[-2].chars) if len(stack) > 1 else self.stack_null

        word_length = len(stack[-1].chars) if len(stack) > 0 else 0
        word_length = Variable(b0.data.new([[word_length]]), requires_grad=False)

        return torch.cat([b0, s0, s1, word_length], -1)


# reference https://www.researchgate.net/profile/Yue_Zhang4/publication/266376262_Character-Level_Chinese_Dependency_Parsing/links/542e18030cf277d58e8e9908/Character-Level-Chinese-Dependency-Parsing.pdf
class ArcStandard(nn.Module):
    def __init__(self, input_dim, feature_dim, num_relation, num_pos, dropout=0.):
        super(ArcStandard, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim

        #self.encoder = QRNN(input_dim, feature_dim, 1,
        #                    window_sizes=3, dropout=dropout)
        self.transition_classifier = TransitionClassifier(input_dim, Transitions.max_len)

        self.num_relation = num_relation
        self.relation_classifier = RelationClassifier(input_dim, num_relation)

        self.num_pos = num_pos
        self.pos_classifier = POSClassifier(input_dim, num_pos)


    def _buffer(self, sentences):
        if isinstance(sentences, PackedSequence):
            buffers, lengths = pad_packed_sequence(sentences, batch_first=False)
        else:
            raise 'buffers must be PackedSequence'

        buffers = [list(torch.split(b.squeeze(1)[:length], 1, 0))
                   for b, length in zip(torch.split(buffers, 1, 1), lengths)]

        buffers = [list(reversed(b)) for b in buffers]

        return buffers, lengths


    def loss(self, sentences, gold):
        '''
        :param sentences: [length, batch, dim]
        :param gold_transitions: [length, batch]
        :return:
        '''
        gold_transitions, gold_relations, gold_pos = gold

        #sentences, _ = self.encoder(sentences)
        State = namedtuple('State', ['nodes', 'buffer', 'stack', 'gold_pos'])

        buffers, lengths = self._buffer(sentences)

        batch_size = len(lengths)
        # [batch_size * [stack_size * [node]]]
        stacks = [[]] * batch_size

        states = [State([], buffer, stack, list(reversed(pos)))
                  for buffer, stack, pos in zip(buffers, stacks, gold_pos)]

        num_action = gold_transitions.size(0)


        transition_losses = []
        transition_correct = 0
        transition_count = 1e-5

        pos_loss = [0]
        pos_correct = 0
        pos_count = 1e-5

        seg_loss = []
        seg_correct = 0
        seg_count = 1e-5

        for t in range(num_action):
            state_t, transition_t, mask_t = [], [], []
            for state, action, last_action in zip(states, gold_transitions[t].data, gold_transitions[t-1].data if t > 0 else [-1]*batch_size):
                if action >= 0:
                    state_t.append(state)
                    transition_t.append(action)
                    mask_t.append(self.mask(len(state.buffer), len(state.stack), last_action))
            if len(transition_t) == 0:
                break

            transition_pred = self.transition_classifier([state.buffer for state in state_t], [state.stack for state in state_t])

            action_mask = torch.cat(mask_t, 0)

            transition_pred = transition_pred * action_mask

            transition_t = Variable(torch.LongTensor(transition_t), requires_grad=False)
            transition_losses.append(F.cross_entropy(transition_pred,
                                               transition_t,
                                               size_average=False))

            action_argmax = transition_pred.max(1)[1]
            transition_correct += (action_argmax.data == transition_t.data).sum()
            transition_count += transition_t.nelement()

            update_nodes, update_contexts = [], []
            pos_words, pos_contexts, pos_list = [], [], []
            for state, action in zip(state_t, transition_t.data):
                if action == Transitions.SHIFT:
                    next_char = state.buffer.pop()
                    new_node = Node(len(state.nodes), [next_char])
                    state.stack.append(new_node)
                    state.nodes.append(new_node)
                elif action == Transitions.APPEND:
                    next_char = state.buffer.pop()
                    partial_word = state.stack[-1]
                    partial_word.chars.append(next_char)
                elif action == Transitions.ARC_LEFT:
                    first = state.stack.pop()
                    second = state.stack.pop()
                    first.lefts.append(second)
                    second.parent_index = first.index
                    state.stack.append(first)
                elif action == Transitions.ARC_RIGHT:
                    first = state.stack.pop()
                    second = state.stack.pop()
                    second.rights.append(first)
                    first.parent_index = second.index
                    state.stack.append(second)

        return (sum(transition_losses), transition_correct, transition_count), (sum(pos_loss)/pos_count, pos_correct, pos_count)

    def mask(self, buffer_len, stack_len, last_action):

        masked = Variable(torch.FloatTensor([[1] * Transitions.max_len]))
        if buffer_len == 0 or stack_len == 0 or last_action == Transitions.ARC_LEFT or last_action == Transitions.ARC_RIGHT:
            masked[0, Transitions.APPEND] = 0.

        if buffer_len == 0:
            masked[0, Transitions.SHIFT] = 0.

        if stack_len < 2:
            masked[0, Transitions.ARC_LEFT] = 0.
            masked[0, Transitions.ARC_RIGHT] = 0.

        return masked

    def parse(self, sentences, beam_size=10):

        #sentences, _ = self.encoder(sentences)

        buffers, lengths = self._buffer(sentences)

        return [self.parse_one(buffer[0:length], beam_size) for buffer, length in zip(buffers, lengths)]

    def parse_one(self, sentence, beam_size=10):

        class State:
            def __init__(self, nodes, buffer, stack, actions, score):
                self.nodes = nodes
                self.buffer = buffer
                self.stack = stack
                self.actions = actions
                self.score = score

            def __deepcopy__(self, memodict={}):

                nodes = copy.deepcopy(self.nodes, memodict)
                buffer = copy.copy(self.buffer)
                stack = copy.deepcopy(self.stack, memodict)
                actions = copy.copy(self.actions)
                score = copy.copy(self.score)

                new_state = State(nodes, buffer, stack, actions, score)
                memodict[id(self)] = new_state
                return new_state


        length = len(sentence) * 2 - 1

        topK = []
        topK.append(State([], sentence, [], [], 0.0))

        next1 = []

        for step in range(length):
            action_pred = self.transition_classifier([state.buffer for state in topK], [state.stack for state in topK])

            action_mask = torch.cat([self.mask(len(state.buffer),
                                              len(state.stack),
                                              state.actions[-1] if len(state.actions) else -1) for state in topK], 0)

            action_pred = action_pred * action_mask
            action_log_prob = F.log_softmax(action_pred).split(1, 0)

            def check(buffer, stack, prev_transitions, transition):

                prev_transition = prev_transitions[-1] if len(prev_transitions) else None

                if transition == Transitions.SHIFT:
                    if len(buffer) == 0:
                        return False
                elif transition == Transitions.APPEND:
                    if prev_transition != Transitions.SHIFT and prev_transition != Transitions.APPEND:
                        return False
                    if len(buffer) == 0 or len(stack) == 0:
                        return False

                    if len(stack) > 0 and len(stack[-1].chars) >= 8:
                        return False
                elif transition == Transitions.ARC_LEFT or transition == Transitions.ARC_RIGHT:
                    if len(stack) < 2:
                        return False
                return True

            step1, step2 = [], []
            for state, log_probs in zip(topK, action_log_prob):
                for action, log_prob in enumerate(log_probs.squeeze().data):
                    if check(state.buffer, state.stack, state.actions, action):
                        if action in [Transitions.SHIFT, Transitions.ARC_LEFT, Transitions.ARC_RIGHT]:
                            step1.append((state, action, state.score+log_prob))
                        else:
                            # append = shift + reduce
                            step2.append((state, action, state.score+log_prob*2))


            sorted_cands = sorted(step1 + next1, key=lambda c: c[2], reverse=True)

            def cand2state(state, action, score):
                # todo optimize
                new_state = copy.deepcopy(state)
                new_state.actions.append(action)
                new_state.score = score

                if action == Transitions.SHIFT:
                    next_char = new_state.buffer.pop()
                    new_node = Node(len(new_state.nodes), [next_char])
                    new_state.stack.append(new_node)
                    new_state.nodes.append(new_node)
                elif action == Transitions.APPEND:
                    next_char = new_state.buffer.pop()
                    partial_word = new_state.stack[-1]
                    partial_word.chars.append(next_char)
                elif action == Transitions.ARC_LEFT:
                    first = new_state.stack.pop()
                    second = new_state.stack.pop()
                    first.lefts.append(second)
                    new_state.stack.append(first)
                elif action == Transitions.ARC_RIGHT:
                    first = new_state.stack.pop()
                    second = new_state.stack.pop()
                    second.rights.append(first)
                    new_state.stack.append(second)

                return new_state

            topK = [cand2state(*c) for c in sorted_cands[0: beam_size]]
            next1 = step2

        return topK[0].actions




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
                          loader.relation_dict, loader.pos_dict,
                          self.hidden_dim,
                          self.dropout,
                          self.shared_weight_decay,
                          self.task_weight_decay)


class ParserTask(Task):
    def __init__(self, name, encoder, vocab,
                 relation_dict, pos_dict,
                 hidden_dim,
                 dropout,
                 shared_weight_decay,
                 task_weight_decay):
        super(ParserTask, self).__init__()

        self.name = name
        self.vocab = vocab
        self.relation_dict = relation_dict
        self.pos_dict = pos_dict
        self.shared_encoder = encoder

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.shared_weight_decay = shared_weight_decay
        self.task_weight_decay = task_weight_decay

        self.parser = ArcStandard(self.shared_encoder.output_dim(),
                                  self.hidden_dim,
                                  len(self.relation_dict),
                                  len(self.pos_dict),
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

    def parse(self, sentences):
        sentences, gazetteers = sentences

        features = self.shared_encoder(sentences, gazetteers)

        return self.parser.parse(features)

    def sample(self, batch_data, use_cuda=False):

        if use_cuda:
            batch_data = self._to_cuda(batch_data)

        sentences, reference = batch_data
        pred_transitions = self.parse(sentences)
        sentences, lengths = pad_packed_sequence(sentences[0], batch_first=True)
        sentences = [sentence[0, :length] for sentence, length in zip(sentences.split(1, 0), lengths)]
        return [UDTree.create(self.vocab.get_word(sentence.data.numpy()), actions).to_line()
                for sentence, actions in zip(sentences, pred_transitions)]


    def evaluation(self, data, use_cuda=False):

        seg_correct = 0
        seg_gold_count = 1e-3
        seg_pred_count = 1e-3
        for batch in data:
            if use_cuda:
                batch = self._to_cuda(batch)

            sentences, (transitions, _, _) = batch
            pred_trans = self.parse(sentences)

            def eval(sentence, gold, pred):

                gold_tree = UDTree.create(sentence, gold)
                pred_tree = UDTree.create(sentence, pred)

                gold_words = gold_tree.get_words()
                pred_words = pred_tree.get_words()

                from collections import Counter


        seg_prec = seg_correct / seg_pred_count
        seg_recall = seg_correct / seg_gold_count
        seg_f = 2*(seg_prec*seg_recall)/(seg_prec+seg_recall+1e-3)
        return {'seg prec':seg_prec, 'seg recall':seg_recall, 'seg F-score':seg_f}


class CTBParseData(Loader):
    def __init__(self, train_paths, test_paths):
        self.train_data = self.load(train_paths)

        self.word_counts = defaultdict(int)
        self.pos_counts = defaultdict(int)
        self.relation_counts = defaultdict(int)

        for chars, (_, relations, pos) in self.train_data:
            for p in pos:
                self.pos_counts[p] += 1

            for r in relations:
                self.relation_counts[r] += 1

            for char in chars:
                self.word_counts[char] += 1
        self.test_data = self.load(test_paths)

        self.train_data = sorted(self.train_data, key=lambda item: len(item[0]), reverse=True)
        self.test_data = sorted(self.test_data, key=lambda item: len(item[0]), reverse=True)

        self.pos_dict = Vocab(list(self.pos_counts.keys()))
        self.relation_dict = Vocab(list(self.relation_counts.keys()))

    def load(self, paths):
        data = []
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

                        data.append(UDTree.parse_stanford_format(words).linearize())
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
                sentences[:sen_len, id] = torch.from_numpy(vocab.convert(words))
                gazes[0:sen_len, id] = torch.from_numpy(
                    np.concatenate([gazetteer.convert(words) for gazetteer in gazetteers], -1))
                tran_len = len(trans)
                transitions[:tran_len, id] = torch.LongTensor(trans)
                relations[:len(rel), id] = torch.LongTensor(self.relation_dict.convert(rel))
                pos[:len(str_pos), id] = torch.LongTensor(self.pos_dict.convert(str_pos))
            yield ((pack_padded_sequence(Variable(sentences), sen_lens),
                    pack_padded_sequence(Variable(gazes), sen_lens)),
                   (Variable(transitions), Variable(relations), Variable(pos)))

    def batch_train(self, vocab, gazetteers, batch_size):
        return self._batch(self.train_data, vocab, gazetteers, batch_size)

    def batch_test(self, vocab, gazetteers, batch_size):
        return self._batch(self.test_data, vocab, gazetteers, batch_size)


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

        self.shared_encoder_config = EncoderConfig()
        self.ctb_parser_config = ParserConfig('ctb_parser',
                                              [os.path.join(self.data_root, 'parser/ctb.dep.train')],
                                              [os.path.join(self.data_root, 'parser/ctb.dep.gold')])

        self.task_configs = [self.ctb_parser_config]

        self.valid_size = 1000//self.batch_size

        # 字属性字典
        self.char_attr = './dict/word-type'
        # 词集合
        self.wordset = {}
        self.model_prefix = 'model/model'

        self.eval_step = 500

        self.epoches = 10

'''
config = MultiTaskConfig()
encoder_config = config.encoder_config
parser_config = config.ctb_parser_config

parser_data = CTBParseData(parser_config.train_paths, parser_config.test_paths)

vocab = Vocab.build(parser_data.word_counts, encoder_config.max_vocab_size)

char2attr = CharacterAttribute.load(config.char_attr)
gazetteers = [char2attr] + [Gazetteer.load(name, path) for name, path in config.wordset]

encoder = Encoder(len(vocab), gazetteers, encoder_config.embedding_dim,
                  encoder_config.hidden_mode, encoder_config.hidden_dim,
                  encoder_config.num_hidden_layer, encoder_config.window_sizes,
                  encoder_config.dropout)

task = ParserTask('ctb_parser', encoder,
                  vocab, parser_data.relation_dict, parser_data.pos_dict,
                  config.ctb_parser_config)

train_data = list(parser_data.batch_train(vocab, gazetteers, config.batch_size))
test_data = list(parser_data.batch_test(vocab, gazetteers, config.batch_size))

valid_data = []
train_data, valid_data = train_test_split(train_data, test_size=1000//config.batch_size)

optimzer = torch.optim.Adam(task.params, lr=1e-2)

for epoch in range(10):

    total_loss = 0.
    total_correct = 0.
    total_count = 0
    for batch_id, batch in enumerate(train_data, start=1):
        task.train()
        task.zero_grad()

        loss, count = task.loss(batch)
        total_loss += loss.data[0] * count
        total_count += count

        loss.backward()
        optimzer.step()


        if batch_id % 200 == 0:
            valid_loss = 0.
            valid_count = 0
            task.eval()
            for valid_batch in valid_data:
                loss, count = task.loss(valid_batch)

                valid_loss += loss.data[0] * count
                valid_count += count

            print('train loss=%f\tvalid loss=%f' % (total_loss / total_count, valid_loss / valid_count))

            total_loss = 0.
            total_correct = 0.
            total_count = 0

            #sample
            import random
            sample = valid_data[random.randint(0, len(valid_data)-1)]
            task.eval()
            print('\n'.join(task.sample(sample)))

'''
