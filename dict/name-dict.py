# encoding='gbk'

import requests
import os
from bs4 import BeautifulSoup
from tqdm import tqdm

data_dir = 'name-dict'

if os.path.exists(data_dir) is False:
    os.mkdir(data_dir)

url_format = 'https://www.babble.com/baby-names/filter/raw/any/any/any/any/page/%d/'
for id in tqdm(range(284, 286)):
    url = url_format % id
    try:
        data = requests.get(url)
        if data.status_code == 200:
            soup = BeautifulSoup(data.content, 'html.parser', from_encoding='utf-8')
            with open('/'.join([data_dir, str(id)]), 'w') as f:
                f.write(str(soup))

    except Exception as e:
        print('%s failed. exception: %s' % (url, str(e)))