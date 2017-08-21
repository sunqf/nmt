


# people2014

import os

'''
path = '/Users/sunqf/startup/quotesbot/nlp-data/chinese_segment/2014'

with open("people2014.txt", 'w') as dest:
    for subdir in os.listdir(path):
        for file in os.listdir(os.path.join(path, subdir)):
            with open(os.path.join(path, subdir, file)) as f:
                for line in f:
                    dest.write(' '.join([item.rsplit('/', 1)[0].replace('[', '[ ') for item in line.replace(']', ' ]').split()]))
                    dest.write('\n')

'''
'''
    The following is a lit of files that are double-annotated and can be
    regarded as gold standard files.
    
    CTB-1 (69 files, 22,316 words)
    chtb_001.fid - chtb_043.fid
    chtb_144.fid - chtb_169.fid
    
    CTB-3 (32 files, 12,027 words)
    chtb_900.fid - chtb_931.fid
    
    CTB-4 (7 files, 13,828 words)
    chtb_1018.fid
    chtb_1020.fid
    chtb_1036.fid
    chtb_1044.fid
    chtb_1060.fid
    chtb_1061.fid
    chtb_1072.fid
    
    CTB-5 (6 files, 15,052 words)
    chtb_1118.fid
    chtb_1119.fid
    chtb_1132.fid
    chtb_1141.fid
    chtb_1142.fid
    chtb_1148.fid
    
    Total: 114 files, 63,223 words (12.46% of the corpus)
'''
ctb_path = '/Users/sunqf/startup/quotesbot/nlp-data/chinese_segment/ctb8.0/data/segmented'
gold_file = [
    'chtb_1018.mz.seg', 'chtb_1020.mz.seg', 'chtb_1036.mz.seg',
    'chtb_1044.mz.seg', 'chtb_1060.mz.seg', 'chtb_1061.mz.seg', 'chtb_1072.mz.seg',
    'chtb_1118.mz.seg', 'chtb_1119.mz.seg', 'chtb_1132.mz.seg',
    'chtb_1141.mz.seg', 'chtb_1142.mz.seg', 'chtb_1148.mz.seg',
]
for i in range(1, 44, 1):
    gold_file.append('chtb_%03d.nw.seg' % i)

for i in range(900, 932, 1):
    gold_file.append('chtb_%03d.nw.seg' %i)

gold_file = set(gold_file)

import re

with open('ctb.train', 'w') as train, open('ctb.gold', 'w') as gold:
    for file in os.listdir(ctb_path):
        print(file)
        with open(os.path.join(ctb_path, file)) as f:
            for line in f:
                if re.match('^<.*> *$', line) is None and len(line.strip()) > 0:
                    if file in gold_file:
                        gold.write(line)
                    else:
                        train.write(line)
