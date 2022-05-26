from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

import urllib
import collections
import os
import zipfile

import numpy as np
import tensorflow as tf
import sys

import vocab_generator as V_gen
import save_embeddings_v2 as S

import random
import re
from sentences_generator import Sentences


def generator(target,context, labels, batch_size):
    batch_target = np.zeros((batch_size, 1))
    batch_context = np.zeros((batch_size, 1))
    batch_labels = np.zeros((batch_size,1))

    while True:
        for i in range(batch_size):
            index= random.randint(0,len(target)-1)
            batch_target[i] = target[index]
            batch_context[i]=context[index]
            batch_labels[i] = labels[index]
        yield [batch_target,batch_context], [batch_labels]        

   
def collect_data(textFile):
    data=[]
    sentences = Sentences(textFile)
    vocab = dict()
    V_gen.build_vocabulary_v2(vocab, sentences)
    for s in sentences:
        for w in s.split(" "):
            w=re.sub("[.,:;'\"!?()]+","",w.lower())
            if w!='':
                data.append(vocab[w])

    return data, vocab            


window_size = 3
vector_dim = 100
epochs = 1000

data,vocab=collect_data(sys.argv[1])
vocab_size=len(vocab)

#sampling_table = sequence.make_sampling_table(vocab_size)
couples, labels = skipgrams(data, vocab_size, window_size=window_size)
word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

dot_product = merge([target, context], mode='dot', dot_axes=1)
dot_product = Reshape((1,))(dot_product)
output = Dense(1, activation='sigmoid')(dot_product)
model = Model(input=[input_target, input_context], output=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['acc'])

print model.summary()

epochs=int(sys.argv[2])



def train(word_target,word_context,labels,epochs):
    arr_1 = np.zeros((1,))
    arr_2 = np.zeros((1,))
    arr_3 = np.zeros((1,))
    for cnt in range(epochs):
        idx = np.random.randint(0, len(labels)-1)
        arr_1[0,] = word_target[idx]
        arr_2[0,] = word_context[idx]
        arr_3[0,] = labels[idx]
        loss = model.train_on_batch([arr_1, arr_2], arr_3)
        if cnt % 100 == 0:
            print("Iteration {}/{}, loss={}".format(cnt, epochs,loss))


train(word_target,word_context,labels,epochs)
            
#model.fit_generator(generator(word_target, word_context,labels,100), steps_per_epoch=100, epochs=epochs)

S.save_embeddings("embedding.txt", embedding.get_weights()[0], vocab)

exit(0)

