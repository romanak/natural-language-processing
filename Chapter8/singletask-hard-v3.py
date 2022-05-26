from __future__ import print_function
import sys
import re
import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten, LSTM, Input, Concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

import io

def vectorizeString(s,lexicon):
    vocabSize = len(lexicon)
    s=str(s)
    result = one_hot(s,round(vocabSize*1.5))
    return result

global ClassLexicon
#ClassLexicon={}



def processLabel(x):
    if x in ClassLexicon:
        return ClassLexicon[x]
    else:
        ClassLexicon[x]=len(ClassLexicon)
        return ClassLexicon[x]



def processLabelOld(x):
    if isinstance(x,int):
        return x
    if isinstance(x,str):
        try:
            r=int(x)
            return r
        except:
            if x in ClassLexicon:
                return ClassLexicon[x]
            else:
                ClassLexicon[x]=len(ClassLexicon)
                return ClassLexicon[x]

            

def loadData(train, test):

    global Lexicon    

    with io.open(train,encoding = "ISO-8859-1") as f:
        trainD = f.readlines()
    f.close()

    with io.open(test,encoding = "ISO-8859-1") as f:
        testD = f.readlines()
    f.close()

    all_text=[]
    for line in trainD:
        m=re.match("^(.+),[^\s]+$",line)
        if m:
            all_text.extend(m.group(1).split(" "))
    for line in testD:
        m=re.match("^(.+),[^\s]+$",line)
        if m:
            all_text.extend(m.group(1).split(" "))
    Lexicon=set(all_text)

    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    
    for line in trainD:
        m=re.match("^(.+),([^\s]+)$",line)
        if m:
            x_train.append(vectorizeString(m.group(1),Lexicon))
            y_train.append(processLabel(m.group(2)))


    for line in testD:
        m=re.match("^(.+),([^\s]+)$",line)
        if m:
            x_test.append(vectorizeString(m.group(1),Lexicon))
            y_test.append(processLabel(m.group(2)))

    return (np.array(x_train),np.array(y_train)),(np.array(x_test),np.array(y_test))
                           
            
    

if __name__=="__main__":

#    train="/data/gender.blogs.tr" # sys.argv[1]
 #   test="/data/gender.blogs.te" # sys.argv[2]

    train="/data/amazon_cells_labelled.tr" # sys.argv[1]
    test="/data/amazon_cells_labelled.te" # sys.argv[2]

    #train="/data/yelp_labelled.tr" # sys.argv[1]
    #test="/data/yelp_labelled.te" # sys.argv[2]

    print('Loading data...')


    ClassLexicon={}
    (x_train,y_train),(x_test,y_test)=loadData(train,test)
    num_classes=len(ClassLexicon)

    epochs = 100
    batch_size=128

 
    #num_classes = max(set([int(y) for y in y_train]))+1
    #print(num_classes, 'classes')

    print('Vectorizing sequence data...')

    max_words=len(Lexicon)+1 #10000

    max_length = 1000 # max(max([len(x) for x in x_train]),max([len(x) for x in x_test]))
    x_train = pad_sequences(x_train, maxlen=max_length, padding='post')
    x_test = pad_sequences(x_test, maxlen=max_length, padding='post')    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print('Building model...')
    inputs=Input(shape=(max_length,))
    x=Embedding(300000, 32)(inputs)
    x=Dense(512,activation='relu')(x)
    x=Flatten()(x) #DA for LSTM
    x=Dense(256, activation='relu')(x)
    x=Dropout(0.5)(x)
    x=Dense(128,activation='relu')(x)
    x=Dropout(0.5)(x)
    y=Dense(num_classes,activation='softmax')(x)

    model=Model(inputs=inputs, outputs=y)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1)
    
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size, verbose=1)

    print(model.metrics_names)

    print(score)
    
    #print('Test score:', score[0])
    #print('Test accuracy:', score[1]
    #)
