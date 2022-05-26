from keras.models import Sequential
from keras.layers import Embedding,Dense,Flatten,Convolution1D
import numpy as np

import vocab_generator as V_gen
import random
import re
from sentences_generator import Sentences
import sys
from keras.preprocessing.sequence import pad_sequences
import save_embeddings_v2 as S



def collect_data(textFile,max_len):
    data=[]
    sentences = Sentences(textFile)
    vocab = dict()
    labels=[]
    V_gen.build_vocabulary_v2(vocab, sentences)
    for s in sentences:
        words=[]
        m=re.match("^([^\t]+)\t(.+)$",s.rstrip())
        if m:
            sentence=m.group(1)
            labels.append(int(m.group(2)))
        for w in sentence.split(" "):
            w=re.sub("[.,:;'\"!?()]+","",w.lower())
            if w!='':
                words.append(vocab[w])
        data.append(words)
    data = pad_sequences(data, maxlen=max_len, padding='post')

    return data,labels, vocab


def collect_test_data(textFile,vocab,max_len):
    data=[]
    sentences = Sentences(textFile)
    labels=[]
    V_gen.build_vocabulary_v2(vocab, sentences)
    for s in sentences:
        words=[]
        m=re.match("^([^\t]+)\t(.+)$",s.rstrip())
        if m:
            sentence=m.group(1)
            labels.append(int(m.group(2)))
        for w in sentence.split(" "):
            w=re.sub("[.,:;'\"!?()]+","",w.lower())
            if w!='':
                if w in vocab:
                    words.append(vocab[w])
                else:
                    words.append(vocab["<unk>"])
        data.append(words)
    data = pad_sequences(data, maxlen=max_len, padding='post')
    return data,labels            


max_len=100
data,labels,vocab=collect_data(sys.argv[1],max_len)
test_data,test_labels=collect_test_data(sys.argv[2],vocab,max_len)

model = Sequential()
embedding=Embedding(len(vocab), 100, input_length=max_len)
model.add(embedding)

model.add(Dense(max_len, activation="relu"))
model.add(Convolution1D(10, 10, padding="same"))
model.add(Dense(max_len, activation="relu"))

model.add(Flatten())
model.add(Dense(1,activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
model.fit(data,labels,epochs=100, verbose=1)



loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
print accuracy

S.save_embeddings("embedding_labeled2.txt", embedding.get_weights()[0], vocab)
