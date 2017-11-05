

from .layer.lm import LanguageModel


from .layer.encoder import Encoder

from .loader import DataLoader
from tqdm import tqdm

import itertools

import os
from sklearn.model_selection import train_test_split
import torch

class Config:

    def __init__(self):
        self.max_vocab_size = 5000
        self.batch_size = 32
        self.embedding_dim = 128
        self.hidden_mode = 'QRNN'
        self.num_hidden_layer = 1
        self.hidden_dim = 128
        self.bidirectional = True
        self.window_sizes = [2, 2]

        self.dropout = 0.3
        self.use_cuda = False

        self.data_root = '/Users/sunqf/startup/quotesbot/nlp-data/chinese_segment/data/'
        #self.data_root = '/home/sunqf/Work/chinese_segment/data'

        self.paths = [os.path.join(self.data_root, 'train/train.all'),
                      *[os.path.join(self.data_root, 'gold', path)
                                  for path in ['bosonnlp/auto_comments.txt', 'bosonnlp/food_comments.txt',
                                        'bosonnlp/news.txt', 'bosonnlp/weibo.txt',
                                        'ctb.gold', 'msr_test_gold.utf8',
                                        'pku_test_gold.utf8']]
                      ]

        # 字属性字典
        self.char_attr = './dict/word-type'
        # 词集合
        self.wordset = {}
        self.model_prefix = 'model/model'

        self.epoch = 10

        self.step_size = 100

config = Config()
loader = DataLoader.build(config.paths, config.char_attr, config.wordset, config.max_vocab_size, False)

data = loader.get_data(config.paths, config.batch_size)

train, valid = train_test_split(list(data), test_size=1000//config.batch_size)

encoder = Encoder(len(loader.vocab), loader.gazetteers, config.embedding_dim,
                  config.hidden_mode, config.hidden_dim, config.num_hidden_layer, config.window_sizes,
                  config.dropout)
lm = LanguageModel(encoder.output_dim()//2, len(loader.vocab), encoder.word_embeds.weight, config.bidirectional, config.dropout)

optimizer = torch.optim.Adam(itertools.chain.from_iterable([encoder.parameters(), lm.parameters()]), weight_decay=1e-5)

for epoch in tqdm(range(config.epoch), desc='epoch', total=config.epoch):
    total_loss = 0.
    total_count = 0
    for id, batch in tqdm(enumerate(train, start=1), desc='batch', total=len(train)):
        sentences, gazetteers, _ = batch
        if len(sentences.batch_sizes) >= 5:
            encoder.train()
            encoder.zero_grad()
            lm.train()
            lm.zero_grad()


            loss, count = lm.criterion(sentences, encoder(sentences, gazetteers))

            total_loss += loss.data[0]
            total_count += count

            avg_loss = loss/count
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm(itertools.chain.from_iterable([encoder.parameters(), lm.parameters()]), 0.25)
            optimizer.step()

        if id % config.step_size == 0:
            encoder.eval()
            lm.eval()

            valid_loss = 0.
            valid_count = 0
            for vbatch in valid:
                sentences, gazetteers, _ = vbatch
                loss, count = lm.criterion(sentences, encoder(sentences, gazetteers))

                valid_loss += loss.data[0]
                valid_count += count

            print('\n\ntrain loss =%0.6f\tvalid loss =%0.6f\n\n'
                  % (total_loss/(total_count+1e-5), valid_loss/(valid_count+1e-5)))

            total_count = 0
            total_loss = 0.