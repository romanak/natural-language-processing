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
        #else:
        #   print("Illegal line:",line)
    for line in testD:
        m=re.match("^(.+),[^\s]+$",line)
        if m:
            all_text.extend(m.group(1).split(" "))
        #else:
        #    print("Illegal line:",line)
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

    train1="/data/amazon_cells_labelled.tr" # sys.argv[1]
    test1="/data/amazon_cells_labelled.te" # sys.argv[2]

    train2="/data/yelp_labelled.tr" # sys.argv[1]
    test2="/data/yelp_labelled.te" # sys.argv[2]

    print('Loading data...')

    ClassLexicon={}
    (x1_train,y1_train),(x1_test,y1_test)=loadData(train1,test1)
    num_classes1=len(ClassLexicon)

    ClassLexicon={}
    (x2_train,y2_train),(x2_test,y2_test)=loadData(train2,test2)
    num_classes2=len(ClassLexicon)

   
    epochs = 50
    batch_size=128

 
    #num_classes = max(set([int(y) for y in y_train]))+1
    #print(num_classes, 'classes')

    print('Vectorizing sequence data...')

#    max_words=len(Lexicon)+1 #10000

    max_length = 1000 # max(max([len(x) for x in x_train]),max([len(x) for x in x_test]))

    x1_train = pad_sequences(x1_train, maxlen=max_length, padding='post')
    y1_train = keras.utils.to_categorical(y1_train, num_classes1)

    x1_test = pad_sequences(x1_test, maxlen=max_length, padding='post')
    y1_test = keras.utils.to_categorical(y1_test, num_classes1)

    x2_train = pad_sequences(x2_train, maxlen=max_length, padding='post')
    y2_train = keras.utils.to_categorical(y2_train, num_classes2)

    x2_test = pad_sequences(x2_test, maxlen=max_length, padding='post')
    y2_test = keras.utils.to_categorical(y2_test, num_classes2)
    

    print('Building model...') # CASE: two inputs (other case: 1 shared input)

    inputsA=Input(shape=(max_length,))
    x1=Embedding(300000, 16)(inputsA)
    x1=Dense(64,activation='relu')(x1)
 #   x1=Dense(32,activation='relu')(x1)
    x1=Flatten()(x1) #DA for LSTM

    inputsB=Input(shape=(max_length,))
    x2=Embedding(300000, 16)(inputsB)
    x2=Dense(64,activation='relu')(x2)
#    x2=Dense(32,activation='relu')(x2)
    x2=Flatten()(x2) #DA for LSTM

    merged = Concatenate()([x1, x2])

#    task1=Dense(256, activation='relu')(merged)
#    task1=Dropout(0.5)(task1)
#    task1=Dense(128,activation='relu')(task1)
#    task1=Dropout(0.5)(task1)

#    task2=Dense(256, activation='relu')(merged) # SHARE
#    task2=Dropout(0.5)(task2)
#    task2=Dense(128,activation='relu')(task2)
#    task2=Dropout(0.5)(task2)

#    y1=Dense(num_classes1,activation='softmax')(task1)
#    y2=Dense(num_classes2,activation='softmax')(task2)
    y1=Dense(num_classes1,activation='softmax')(merged)
    y2=Dense(num_classes2,activation='softmax')(merged)


    model=Model(inputs=[inputsA, inputsB],outputs=[y1,y2])
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit([x1_train,x2_train], [y1_train,y2_train],
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1)
    
    score = model.evaluate([x1_test,x2_test], [y1_test,y2_test],
                           batch_size=batch_size, verbose=1)

    print(model.metrics_names)

    print(score)
    
    #print('Test score:', score[0])
    #print('Test accuracy:', score[1]
    #)
