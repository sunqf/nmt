
import os

class Config:
    def __init__(self):
        self.max_vocab_size = 5000
        self.batch_size = 8
        self.embedding_dim = 512
        self.hidden_mode = 'QRNN'
        self.num_hidden_layer = 2
        self.kernel_sizes = [3, 3]

        self.dropout = 0.3
        self.use_cuda = True

        #self.data_root = '/Users/sunqf/startup/quotesbot/nlp-data/chinese_segment/data/'
        self.data_root = '/home/sunqf/Work/chinese_segment/data'
        self.coarse_train_paths = [os.path.join(self.data_root, 'train/train.all')]
        self.coarse_eval_paths = [os.path.join(self.data_root, 'gold', path)
                                  for path in ['bosonnlp/auto_comments.txt', 'bosonnlp/food_comments.txt',
                                        'bosonnlp/news.txt', 'bosonnlp/weibo.txt',
                                        'ctb.gold', 'msr_test_gold.utf8',
                                        'pku_test_gold.utf8']]

        self.fine_train_paths = [os.path.join(self.data_root, 'train/ctb.train')]
        self.fine_eval_paths = [os.path.join(self.data_root, 'gold/ctb.gold')]

        # 字属性字典
        self.char_attr = './dict/word-type'
        # 词集合
        self.wordset = {}
        self.model_prefix = 'model/model'

        self.lm_weight = 0.5

        self.eval_step = 2000

        self.coarse_epoches = 5
        self.fine_epoches = 5
