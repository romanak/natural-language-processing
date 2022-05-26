from keras.models import Sequential
from keras.layers import Embedding,Dense,Flatten,Convolution1D
import numpy as np
import vocab_generator as V_gen
import random
import re
import sys
from keras.preprocessing.sequence import pad_sequences
import zipfile


def load_embedding(f, vocab, embedding_dimension):
    embedding_index = {}
    f = open(f)
    n=0
    for line in f:
        values = line.split()
        word = values[0]
        if word in vocab: #only store words in current vocabulary
            coefs = np.asarray(values[1:], dtype='float32')
            if n: #skip header line
                embedding_index[word] = coefs
            n+=1
    f.close()

    embedding_matrix = np.zeros((len(vocab) + 1, embedding_dimension))
    for word, i in vocab.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def load_embedding_zipped(f, vocab, embedding_dimension):
    embedding_index = {}
    with zipfile.ZipFile(f) as z:
        with z.open("glove.6B.100d.txt") as f:
            n=0
            for line in f:
                if n:
                    values = line.split()
                    word = values[0]
                    if word in vocab: #only store words in current vocabulary
                        coefs = np.asarray(values[1:], dtype='float32')
                        embedding_index[word] = coefs
                n+=1
    z.close()
    embedding_matrix = np.zeros((len(vocab) + 1, embedding_dimension))
    for word, i in vocab.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix





def getLines(f):
    lines = [line.rstrip() for line in open(f)]
    return lines



def create_vocabulary(vocabulary, sentences):
        vocabulary["<unk>"]=0
	for sentence in sentences:
		for word in sentence.strip().split():
                        word=re.sub("[.,:;'\"!?()]+","",word.lower())
                        if word not in vocabulary:
			        vocabulary[word]=len(vocabulary)


def process_training_data(textFile,max_len):
    data=[]
    sentences = getLines(textFile)
    vocab = dict()
    labels=[]
    create_vocabulary(vocab, sentences)
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


def process_test_data(textFile,vocab,max_len):
    data=[]
    sentences = getLines(textFile)
    labels=[]
    create_vocabulary(vocab, sentences)
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
data,labels,vocab=process_training_data(sys.argv[1],max_len)
test_data,test_labels=process_test_data(sys.argv[2],vocab,max_len)

model = Sequential()

embedding_dimension=100

#embedding_matrix=load_embedding(sys.argv[3],vocab,embedding_dimension)
embedding_matrix=load_embedding_zipped(sys.argv[3],vocab,embedding_dimension)

embedding = Embedding(len(vocab) + 1,
                            embedding_dimension,
                            weights=[embedding_matrix],
                            input_length=max_len,
                            trainable=True)
model.add(embedding)
model.add(Flatten())
model.add(Dense(1,activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['acc'])
model.fit(data,labels,epochs=100, verbose=1)

loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
print accuracy

