import gzip
import os
import sys
import re
from lxml import etree
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

path = sys.argv[1]
files = [os.path.join(root, f)
         for root, dirs, files in os.walk(path)
         for f in files
         if f.lower().endswith('.gz')]

output_path = 'texts/'


def f(fn):
    print(fn)

    output_fn = output_path + fn.replace('../', '').replace('.xml.gz', '.txt')
    d = os.path.dirname(output_fn)
    os.makedirs(d, exist_ok=True)

    with gzip.open(fn, 'r') as f:
        file_content = f.read()
        try:
            tree = etree.fromstring(file_content)
            text = str(etree.tostring(tree, encoding='utf8', method='text'))
            text = text.replace('\\n', ' ')
            text = text.replace("\\'", "'")
            # text = re.sub('\s\s+', ' ', text)
            # text = text.replace(' ,', ',')
            # text = text.replace(' .', '.')
            # text = text.replace(' !', '!')
            # text = text.replace(' ?', '?')
            # text = text.replace('[', '')
            # text = text.replace(']', '')
            # text = text.replace('(', '')
            # text = text.replace(')', '')
            # text = text.replace('" ', '"')
            # text = text.replace(' "', '"')
            # text = re.sub('\.\.+', '.', text)
            # text = re.sub('--+', '-', text)
            text = text.replace('{ y : i }', '')
            text = re.sub('\\\\x\w\w', '', text)
            text = re.sub('\d', ' ', text)
            text = re.sub('[?.,\-!\[\]()"\'{}:]', ' ', text)
            text = re.sub('\s\s+', ' ', text)
            text = re.sub('<.*?>', '', text)
            text = text.lower()

            with open(output_fn, 'w') as f:
                f.write(text)

        except etree.XMLSyntaxError:
            pass


pool = ThreadPool(cpu_count() // 2)
pool.map(f, files)

