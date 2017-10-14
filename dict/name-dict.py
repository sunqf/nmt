# encoding='gbk'

import requests
import os
from bs4 import BeautifulSoup
from tqdm import tqdm

data_dir = 'han-dict'

if os.path.exists(data_dir) is False:
    os.mkdir(data_dir)

url_format = 'http://xh.5156edu.com/html3/%d.html'
for id in tqdm(range(14674, 22526)):
    url = url_format % id
    try:
        data = requests.get(url)
        if data.status_code == 200:
            soup = BeautifulSoup(data.content, 'html.parser', from_encoding='gbk')

            with open('/'.join([data_dir, str(id)]), 'w') as f:
                f.write(str(soup.find('table', style="word-break:break-all")))
    except Exception as e:
        print('%s failed. exception: %s' % (url, str(e)))