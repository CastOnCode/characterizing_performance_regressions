from gensim.test.utils import datapath
from gensim import utils
import os
import tempfile
import gensim.models.word2vec

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    def __iter__(self):
        #corpus_path = datapath('lee_background.cor')
        path_to_dirs = r'git_txts'
        dirs = os.listdir(path_to_dirs)
        corpus_paths = [path_to_dirs+'/'+k+'/' for k in dirs]
        for commit in corpus_paths:
            a = os.listdir(commit)
            for filename in a:
                with open(commit+filename, encoding="utf8") as f:
                    content = f.read()
                yield utils.simple_preprocess(content)

sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences, min_count=5, size=100, workers=4, iter=1)
num_of_epochs = 8
for ep in range(num_of_epochs):
    print('start of epoch N '+str(ep+1))
    model.train(sentences = sentences, total_examples = model.corpus_count, epochs=1)
    model.save('word2vec_21_04_'+str(ep+1)+'.model')

model.save('word2vec_20_04.model')
