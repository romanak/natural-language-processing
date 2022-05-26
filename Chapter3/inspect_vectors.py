import sys
import numpy as np
from gensim.models import KeyedVectors
import gensim.models
import os



w2v = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(os.path.dirname(__file__), 'embedding.txt'), binary=False,unicode_errors='ignore')

#w2v = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(os.path.dirname(__file__), 'cp852_fasttext.vec'), binary=False,unicode_errors='ignore')

#w2v = gensim.models.word2vec.Word2Vec.load_word2vec_format('./embedding.txt', binary=False)

#print w2v.wv.vocab
print w2v.most_similar(sys.argv[1],topn=3)
#print w2v.most_similar(positive=['kidney'])
