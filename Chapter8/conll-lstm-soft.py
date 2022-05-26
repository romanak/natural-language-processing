import numpy
import keras
from keras.datasets import imdb
from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten, LSTM, Input, Concatenate
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras.backend as K
import sys
import numpy as np

global Lex
Lex={}

global ClassLex
ClassLex={}

def lookup(feat):
    if feat not in Lex:
        Lex[feat]=len(Lex)
    return Lex[feat]


def class_lookup(feat):
    if feat not in ClassLex:
        ClassLex[feat]=len(ClassLex)
    return ClassLex[feat]




def load_conll(train, test):
    x1_train=[]
    y1_train=[]
    x1_test=[]
    y1_test=[]
    x2_train=[]
    y2_train=[]
    x2_test=[]
    y2_test=[]
    
    tr=open(train,"r")
    for line in tr:
        if line.rstrip()=='':
            continue
        features=line.rstrip().split("|")
        target=features.pop().split(" ")
        target_word=target[0]
        target_y1=target[1]
        target_y2=target[2]
        
        y1_train.append(class_lookup(target_y1))
        y2_train.append(class_lookup(target_y2))

        l=lookup(target_word)
        x1=[l]
        x2=[l]
        for feature in features:
            if feature=='':
                continue
            feature_split=feature.split(" ")
            x1.append(lookup(feature_split[0]))
            x1.append(lookup(feature_split[1]))
            x2.append(lookup(feature_split[0]))
            x2.append(lookup(feature_split[2]))            
        x1_train.append(x1)
        x2_train.append(x2)
    tr.close()


    te=open(test,"r")
    for line in te:
        if line.rstrip()=='':
            continue
        features=line.rstrip().split("|")
        target=features.pop().split(" ")
        target_word=target[0]
        target_y1=target[1]
        target_y2=target[2]
        
        y1_test.append(class_lookup(target_y1))
        y2_test.append(class_lookup(target_y2))

        l=lookup(target_word)
        x1=[l]
        x2=[l]
        
        
        for feature in features:
            if feature=='':
                continue
            feature_split=feature.split(" ")
            x1.append(lookup(feature_split[0]))
            x1.append(lookup(feature_split[1]))
            x2.append(lookup(feature_split[0]))
            x2.append(lookup(feature_split[2]))
        x1_test.append(x1)
        x2_test.append(x2)
    te.close()

    return (np.array(x1_train), np.array(y1_train)),(np.array(x2_train),np.array(y2_train)),(np.array(x1_test),np.array(y1_test)),(np.array(x2_test),np.array(y2_test))
            
                        

if __name__=="__main__":

    maxwords = 5000

    train="/data/esp.train.csv"
    test="/data/esp.testa.csv"
    
    (x1_train, y1_train), (x2_train, y2_train), (x1_test,y1_test),(x2_test, y2_test)= load_conll(train, test) #sys.argv[1],sys.argv[2])


    
   # num_classes1=max(set(y1_train+y1_test))+1
   # num_classes2=max(set(y2_train+y2_test))+1

    num_classes1=np.max(np.concatenate((y1_train,y1_test),axis=None))+1
    num_classes2=np.max(np.concatenate((y2_train,y2_test),axis=None))+1

    y1_train = keras.utils.to_categorical(y1_train, num_classes1)
    y1_test = keras.utils.to_categorical(y1_test, num_classes1)
    y2_train = keras.utils.to_categorical(y2_train, num_classes2)
    y2_test = keras.utils.to_categorical(y2_test, num_classes2)

    
    num_words=len(Lex)
    embedding_vector_length = 32

    max_length=5
    batch_size=128
    epochs=50


    SINGLETASK={}
    MULTITASK={}
    
    # ===================== SINGLE TASK
    
    inputsA=Input(shape=(max_length,))    
    x1=Embedding(num_words, embedding_vector_length)(inputsA)
    x1=Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x1)
    x1=MaxPooling1D(pool_size=2)(x1)
    x1=LSTM(100)(x1)
    y1=Dense(num_classes1, activation='softmax')(x1)
    model=Model(inputs=inputsA,outputs=y1)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])
    history = model.fit(x1_train, y1_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1)
    score = model.evaluate(x1_test, y1_test,
                           batch_size=batch_size, verbose=1)

    SINGLETASK["task1"]=score[1]
    

    inputsB=Input(shape=(max_length,))    
    x2=Embedding(num_words, embedding_vector_length)(inputsB)
    x2=Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x2)
    x2=MaxPooling1D(pool_size=2)(x2)
    x2=LSTM(100)(x2)
    y2=Dense(num_classes2, activation='softmax')(x2)
    model=Model(inputs=inputsB,outputs=y2)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])
    history = model.fit(x2_train, y2_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1)
    score = model.evaluate(x2_test, y2_test,
                           batch_size=batch_size, verbose=1)

    SINGLETASK["task2"]=score[1]
    

    
    # ================= MULTITASK =========================================

    
    
    inputsA=Input(shape=(max_length,))    
    x_a=Embedding(num_words, embedding_vector_length)(inputsA)
    x1=Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x_a)
    x1=MaxPooling1D(pool_size=2)(x1)
    x1=LSTM(100)(x1)

    inputsB=Input(shape=(max_length,))    
    x_b=Embedding(num_words, embedding_vector_length)(inputsB)
    x2=Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x_b)
    x2=MaxPooling1D(pool_size=2)(x2)
    x2=LSTM(100)(x2)

    y1=Dense(num_classes1, activation='softmax')(x1)
    y2=Dense(num_classes2, activation='softmax')(x2)

    model=Model(inputs=[inputsA, inputsB],outputs=[y1,y2])

    x_a=Flatten()(x_a)
    x_b=Flatten()(x_b)

    def custom_loss(a,b):
        def loss(y_true,y_pred):
            e1=keras.losses.categorical_crossentropy(y_true,y_pred)
            e2=keras.losses.mean_squared_error(a,b)
            e3=1.0-keras.losses.cosine_proximity(a,b)
            e4=K.mean(K.square(a-b), axis=-1)
            return e1+e2+e3+e4
        return loss
    
    
    model.compile(#loss='sparse_categorical_crossentropy',
                  loss=custom_loss(x_a,x_b),
                  optimizer='adam',
                  metrics=['categorical_accuracy'])


    history = model.fit([x1_train,x2_train], [y1_train,y2_train],
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1)

    
    score = model.evaluate([x1_test,x2_test], [y1_test,y2_test],
                           batch_size=batch_size, verbose=1)

    print(model.metrics_names)

    print(score)
    
    MULTITASK["task1+task2"]=(score[3],score[4])

    f=open("conll-results-soft.txt","w")
    f.write("Task1:%f\n"%(SINGLETASK["task1"]))
    f.write("Task2:%f\n"%(SINGLETASK["task2"]))
    f.write("Task1+Task2:%f--%f\n"%(MULTITASK["task1+task2"][0],MULTITASK["task1+task2"][1]))
    f.close()
    
    if SINGLETASK["task1"]<MULTITASK["task1+task2"][0]:
        print("TASK1 improved")
    if SINGLETASK["task2"]<MULTITASK["task1+task2"][1]:
        print("TASK2 improved")
        
        
        

