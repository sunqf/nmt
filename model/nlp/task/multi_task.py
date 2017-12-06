import torch

from ..layer.encoder import Encoder
from .parser3 import ParserConfig
from .tagger import TaggerConfig
from ..util.vocab import Vocab, CharacterAttribute, Gazetteer
from sklearn.model_selection import train_test_split

import random
from tqdm import tqdm


class MultiTask:

    def __init__(self, tasks, task_weights, use_cuda=False):
        super(MultiTask, self).__init__()

        self.use_cuda = use_cuda

        self.tasks = [t.cuda() if use_cuda else t for t in tasks]
        self.task_weights = task_weights

        self.optimizers = [torch.optim.Adam(t.params, lr=2e-3) for t in self.tasks]


    def train(self, train_data, valid_data, eval_data, epoch, eval_step, model_prefix):

        for epoch in tqdm(range(epoch), desc='epoch', total=epoch):
            train_losses = [0.] * len(self.tasks)
            task_step_count = [0] * len(self.tasks)
            for batch_id, (task_id, task_data) in tqdm(enumerate(train_data, start=1),
                                                       desc='batch',
                                                       total=len(train_data)):
                task = self.tasks[task_id]
                task.train()
                task.zero_grad()

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
            print('\n\n')

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
        self.batch_size = 32
        self.data_root = '/Users/sunqf/startup/quotesbot/nlp-data/chinese_segment/data/'
        # self.data_root = '/home/sunqf/Work/chinese_segment/data'

        self.use_cuda = False

        import os

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

        ctb_parser_config = ParserConfig('ctb_parser',
                                         [os.path.join(self.data_root, 'parser/ctb.dep.train')],
                                         [os.path.join(self.data_root, 'parser/ctb.dep.gold')])

        self.task_configs = [# people2014,
                             ctb, msr, pku, nlpcc, ctb_pos,
                             ctb_parser_config]

        self.valid_size = 1000 // self.batch_size

        # 字属性字典
        self.char_attr = './dict/word-type'
        # 词集合
        self.wordset = {}
        self.model_prefix = 'model/model'

        self.eval_step = 500

        self.epoches = 10

def build(multi_config):

    # load all dataset
    loaders = [config.loader() for config in multi_config.task_configs]

    # build shared vocab
    import collections
    word_counts = collections.defaultdict(int)
    for loader in loaders:
        word_counts.update(loader.word_counts)
    shared_vocab = Vocab.build(word_counts, multi_config.encoder_config.max_vocab_size)

    # compute task weight by word count
    task_sizes = [sum(loader.word_counts.values()) for loader in loaders]
    min_sizes = min(task_sizes)
    task_weights = [float(min_sizes)/s for s in task_sizes]

    # load gazetters
    char2attr = CharacterAttribute.load(multi_config.char_attr)
    gazetteers = [char2attr] + [Gazetteer.load(name, path) for name, path in multi_config.wordset]

    # merge all data
    train_data = []
    valid_data = []
    eval_data = []

    for task_id, loader in enumerate(loaders):
        temp_train, temp_valid = train_test_split(list(loader.batch_train(shared_vocab, gazetteers, multi_config.batch_size)),
                                                  test_size=multi_config.valid_size)

        train_data += zip([task_id] * len(temp_train), temp_train)
        valid_data += zip([task_id] * len(temp_valid), temp_valid)

        temp_eval = list(loader.batch_test(shared_vocab, gazetteers, multi_config.batch_size))
        if len(temp_eval) > 0:
            eval_data.append((task_id, temp_eval))

    random.shuffle(train_data)
    random.shuffle(valid_data)

    encoder_config = multi_config.encoder_config
    shared_encoder = Encoder(len(shared_vocab), gazetteers,
                             encoder_config.embedding_dim,
                             encoder_config.hidden_mode,
                             encoder_config.hidden_dim,
                             encoder_config.num_hidden_layer,
                             encoder_config.window_sizes,
                             encoder_config.dropout)

    tasks = [config.create_task(shared_vocab, shared_encoder) for config in multi_config.task_configs]

    return MultiTask(tasks, task_weights, multi_config.use_cuda), (train_data, valid_data, eval_data)


multi_config = MultiTaskConfig()

print('build stage')
multi_task, (train_data, valid_data, eval_data) = build(multi_config)

print('train stage')
multi_task.train(train_data, valid_data, eval_data, multi_config.epoches, multi_config.eval_step, multi_config.model_prefix)