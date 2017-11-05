
import os
import collections

class Config:
    def __init__(self):
        self.max_vocab_size = 5000
        self.batch_size = 16
        self.embedding_dim = 64
        self.hidden_mode = 'QRNN'
        self.num_hidden_layer = 2
        self.hidden_dim = 128
        self.window_sizes = [2, 2]

        self.dropout = 0.3
        self.use_cuda = True

        self.data_root = '/Users/sunqf/startup/quotesbot/nlp-data/chinese_segment/data/'
        #self.data_root = '/home/sunqf/Work/chinese_segment/data'
        self.coarse_train_paths = [os.path.join(self.data_root, 'train/train.all')]
        self.coarse_eval_paths = [os.path.join(self.data_root, 'gold', path)
                                  for path in ['bosonnlp/auto_comments.txt', 'bosonnlp/food_comments.txt',
                                        'bosonnlp/news.txt', 'bosonnlp/weibo.txt',
                                        'ctb.gold', 'msr_test_gold.utf8',
                                        'pku_test_gold.utf8']]

        #self.fine_train_paths = [os.path.join(self.data_root, 'train/ctb.train')]
        #self.fine_eval_paths = [os.path.join(self.data_root, 'gold/ctb.gold')]

        self.fine_train_paths = [os.path.join(self.data_root, 'pos/ctb.pos.train')]
        self.fine_eval_paths = [os.path.join(self.data_root, 'pos/ctb.pos.gold')]

        # 字属性字典
        self.char_attr = './dict/word-type'
        # 词集合
        self.wordset = {}
        self.model_prefix = 'model/model'

        self.lm_weight = 0.5

        self.eval_step = 2000

        self.coarse_epoches = 5
        self.fine_epoches = 5


class TaggerConfig:

    def __init__(self,
                 name,
                 train_paths, eval_paths, with_type,
                 hidden_model='QRNN', hidden_dim=64, window_sizes=[2,2],
                 bidirectional=True,
                 dropout=0.3):
        self.name = name
        self.train_paths = train_paths
        self.eval_paths = eval_paths
        self.with_type = with_type

        self.hidden_model = hidden_model
        self.hidden_dim = hidden_dim
        self.window_sizes = window_sizes
        self.bidirectional = bidirectional
        self.dropout = dropout


class MultiTaskConfig:

    def __init__(self):
        self.max_vocab_size = 5000
        self.batch_size = 32
        self.embedding_dim = 128
        self.hidden_mode = 'QRNN'
        self.num_hidden_layer = 1
        self.hidden_dim = 128
        self.window_sizes = [2, 2]

        self.dropout = 0.3
        self.use_cuda = False

        self.data_root = '/Users/sunqf/startup/quotesbot/nlp-data/chinese_segment/data/'
        #self.data_root = '/home/sunqf/Work/chinese_segment/data'

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

        self.tasks = [people2014, ctb, msr, pku, nlpcc, ctb_pos]

        self.test_size = 1000//self.batch_size

        # 字属性字典
        self.char_attr = './dict/word-type'
        # 词集合
        self.wordset = {}
        self.model_prefix = 'model/model'

        self.eval_step = 500

        self.epoches = 10