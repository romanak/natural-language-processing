import argparse
import numpy as np
from keras import backend as K
from keras.optimizers import RMSprop
from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Lambda, LSTM, Dropout, BatchNormalization, Activation
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Convolution1D



from segmentAndVectorizeDocuments_v2 import *

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random


def splitData(X,y, max_samples_per_author=10):
    X,y=shuffle(X,y,random_state=42)
    AuthorsX={}

    for (x,y) in zip(X,y):
            y=np.where(y==1)[0][0]
            if y in AuthorsX:
                    AuthorsX[y].append(x)
            else:
                    AuthorsX[y]=[x]

#    max_samples_per_author=10

    X_left=[]
    X_right=[]
    y_lr=[]

    Done={}
    for author in AuthorsX:
        nb_texts=len(AuthorsX[author])
        nb_samples=min(nb_texts, max_samples_per_author)
        left_docs=np.array(AuthorsX[author])
        random_indexes=np.random.choice(left_docs.shape[0], nb_samples, replace=False)        
        left_sample=np.array(AuthorsX[author])[random_indexes]
        for other_author in AuthorsX:
            if  (other_author,author) in Done:
                    pass
            Done[(author,other_author)]=1
            
            right_docs=np.array(AuthorsX[other_author])
            
            nb_samples_other=min(len(AuthorsX[other_author]), max_samples_per_author)
            random_indexes_other=np.random.choice(right_docs.shape[0], nb_samples_other, replace=False)            
            right_sample=right_docs[random_indexes_other]
            
            for (l,r) in zip(left_sample,right_sample):
                    X_left.append(l)
                    X_right.append(r)            
                    if author==other_author:
                            y_lr.append(1.0)
                    else:
                            y_lr.append(0.0)
    return np.array(X_left),np.array(X_right),np.array(y_lr)



    
def exponent_neg_manhattan_distance(left, right):
        return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))



if __name__=="__main__":
        
    #train="/data/pan12-authorship-attribution-training-corpus-2012-03-28/"
    #test="/data/pan12-authorship-attribution-test-corpus-2012-05-24/GT/"

    train=sys.argv[1]
    test=sys.argv[2]
    
    nb_epochs=5

    (tokenizer, labelHash)=createTokenizer(train,test)

    input_dim = 500 # word chunks

    nb_lstm_units=10
    
    X, y=vectorizeDocumentsBOW(train,tokenizer,labelHash,input_dim)

    nb_classes = len(labelHash)
    vocab_size=len(tokenizer.word_index)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    X_train_left, X_train_right, y_train_lr=splitData(X_train,y_train,20)
    X_test_left, X_test_right, y_test_lr=splitData(X_test,y_test,20)


    left_input = Input(shape=(input_dim,), dtype='int32')
    right_input = Input(shape=(input_dim,), dtype='int32')

    embedding_layer = Embedding(vocab_size, 300, input_length=input_dim)


    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    shared_lstm = LSTM(nb_lstm_units)

    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)


    model_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

    model = Model([left_input, right_input], [model_distance])

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.fit([X_train_left, X_train_right], y_train_lr, batch_size=64, nb_epoch=nb_epochs,
                            validation_split=0.3, verbose=2)
    model.evaluate([X_test_left, X_test_right], y_test_lr)
    
    loss, accuracy = model.evaluate([X_test_left, X_test_right], y_test_lr, verbose=0)
    print('Accuracy: %f' % (accuracy*100))
