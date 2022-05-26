from segmentAndVectorizeDocuments import *
import sys
import re

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution1D
from keras.layers.embeddings import Embedding

import pandas as pd
import numpy as np


if __name__=="__main__":

    (tokenizer, labelHash)=createTokenizer(sys.argv[1],sys.argv[2])

    input_dim = 300 # word chunks
    
    X_train, y_train=vectorizeDocumentsBOW(sys.argv[1],tokenizer,labelHash,input_dim)
    X_test, y_test=vectorizeDocumentsBOW(sys.argv[2], tokenizer, labelHash,input_dim)

    nb_classes = len(labelHash)

    model = Sequential()
    model.add(Embedding(25000, 300, input_length=300))
    model.add(Dense(300, activation='relu'))
    model.add(Convolution1D(32, 30, padding="same"))
    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.3, verbose=2)   
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy: %f' % (accuracy*100))
    

