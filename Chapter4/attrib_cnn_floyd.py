from segmentAndVectorizeDocuments_v2 import *
import sys
import re

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Convolution1D
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle



if __name__=="__main__":

    train="/data/pan12-authorship-attribution-training-corpus-2012-03-28/"
    test="/data/pan12-authorship-attribution-test-corpus-2012-05-24/GT/"
    
    nb_epochs=10

    (tokenizer, labelHash)=createTokenizer(train,test)

    input_dim = 500 # word chunks
    
  #  X_train, y_train=vectorizeDocumentsBOW(train,tokenizer,labelHash,input_dim)

    X, y=vectorizeDocumentsBOW(train,tokenizer,labelHash,input_dim)

    #X,y=shuffle(X,y,random_state=42)

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
#    X_test, y_test=vectorizeDocumentsBOW(test, tokenizer, labelHash,input_dim)
    
    nb_classes = len(labelHash)
    vocab_size=len(tokenizer.word_index)

    model = Sequential()
    model.add(Embedding(vocab_size, 300, input_length=input_dim))
    model.add(Dense(300, activation='relu'))
    model.add(Convolution1D(32, 30, padding="same"))
    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])

    print model.summary()
    
    model.fit(X_train, y_train, epochs=nb_epochs, shuffle=True,batch_size=16, validation_split=0.3, verbose=2)

#    model.fit(X_train, y_train, epochs=nb_epochs, shuffle=True,batch_size=64, verbose=2,validation_data=(X_test, y_test))
        
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy: %f' % (accuracy*100))


