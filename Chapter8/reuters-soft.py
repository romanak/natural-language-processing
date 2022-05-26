from __future__ import print_function
import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten, LSTM, Input, Concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K

#https://github.com/keras-team/keras/issues/12072
reutersTopics= ['cocoa','grain','veg-oil','earn','acq','wheat','copper','housing','money-supply',
   'coffee','sugar','trade','reserves','ship','cotton','carcass','crude','nat-gas',
   'cpi','money-fx','interest','gnp','meal-feed','alum','oilseed','gold','tin',
   'strategic-metal','livestock','retail','ipi','iron-steel','rubber','heat','jobs',
   'lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi','hog','lead']

# Take first 10 topics=> 10*(10-1)/2 = 45 pairs.

max_words = 1000
batch_size = 32
epochs = 30

print('Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,test_split=0.1)


global ClassLexicon
ClassLexicon={}

def processLabel(x):
    if x in ClassLexicon:
        return ClassLexicon[x]
    else:
        ClassLexicon[x]=len(ClassLexicon)
        return ClassLexicon[x]

    

Stored={}
for x in range(46):
    for y in range(46):
        if x==y:
            continue
        if (x,y) not in Stored and (y,x) not in Stored:
            Stored[(x,y)]=1

Tried={}


SINGLESCORES={}
MULTISCORES={}
SUCCES={}


for (topic1,topic2) in Stored:
    for (topic3,topic4) in Stored:
        if (topic1,topic2)==(topic3,topic4):
            continue
        if topic1 in (topic3,topic4) or topic2 in (topic3,topic4):
            continue
        if (topic1,topic2) in Tried or (topic3,topic4) in Tried:
            continue
        Tried[(topic1,topic2)]=1
        Tried[(topic3,topic4)]=1
        
        ClassLexicon={}
        ClassLexicon[topic1]=ClassLexicon[topic2]=0
        ClassLexicon[topic3]=ClassLexicon[topic4]=1
        
        
        indices_train1=[i for i in range(len(y_train)) if y_train[i] in [topic1,topic2]]
        indices_test1=[i for i in range(len(y_test)) if y_test[i] in [topic1,topic2]]        
        indices_train2=[i for i in range(len(y_train)) if y_train[i] in [topic3,topic4]]
        indices_test2=[i for i in range(len(y_test)) if y_test[i] in [topic3,topic4]]
        
        x1_train=np.array([x_train[i] for i in indices_train1])
        y1_train=np.array([processLabel(y_train[i]) for i in indices_train1])

        ClassLexicon={}
                
        x1_test=np.array([x_test[i] for i in indices_test1])
        y1_test=np.array([processLabel(y_test[i]) for i in indices_test1])      
        
        ClassLexicon={}
        
        x2_train=np.array([x_train[i] for i in indices_train2])
        y2_train=np.array([processLabel(y_train[i]) for i in indices_train2])

        ClassLexicon={}
        
        x2_test=np.array([x_test[i] for i in indices_test2])
        y2_test=np.array([processLabel(y_test[i]) for i in indices_test2])
        
        num_classes1=2 #len(set(y1_train))
        num_classes2=2 #len(set(y2_train))
        max_length=1000

        x1_train = pad_sequences(x1_train, maxlen=max_length, padding='post')
        y1_train = keras.utils.to_categorical(y1_train, num_classes1)
        x1_test = pad_sequences(x1_test, maxlen=max_length, padding='post')
        y1_test = keras.utils.to_categorical(y1_test, num_classes1)
        x2_train = pad_sequences(x2_train, maxlen=max_length, padding='post')
        y2_train = keras.utils.to_categorical(y2_train, num_classes2)
        x2_test = pad_sequences(x2_test, maxlen=max_length, padding='post')
        y2_test = keras.utils.to_categorical(y2_test, num_classes2)

        
        if len(x1_train)<300 or len(x2_train)<300:
            continue


        min_train=min(len(x1_train),len(x2_train))
        x1_train=x1_train[:min_train]
        x2_train=x2_train[:min_train]
        y1_train=y1_train[:min_train]
        y2_train=y2_train[:min_train]
        
        min_test=min(len(x1_test),len(x2_test))
        x1_test=x1_test[:min_test]
        x2_test=x2_test[:min_test]
        y1_test=y1_test[:min_test]
        y2_test=y2_test[:min_test]



        #===================================== SINGLE TASK =============================
        
        print('Building model for topics %s+%s'%(reutersTopics[topic1], reutersTopics[topic2]))
        inputs=Input(shape=(max_length,))
        x=Embedding(300000, 16)(inputs)
        x=Dense(64,activation='relu')(x)
        x=Flatten()(x) #DA for LSTM
        y=Dense(num_classes1,activation='softmax')(x)

        model=Model(inputs=inputs, outputs=y)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        history = model.fit(x1_train, y1_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=0,
                            validation_split=0.1)
        
        score = model.evaluate(x1_test, y1_test,
                               batch_size=batch_size, verbose=1)
        
        print(model.metrics_names)
        print(score)
        SINGLESCORES[(topic1,topic2)]=score[1]


        print('Building model for topics %s+%s'%(reutersTopics[topic3], reutersTopics[topic4]))
        inputs=Input(shape=(max_length,))
        x=Embedding(300000, 16)(inputs)
        x=Dense(64,activation='relu')(x)
        x=Flatten()(x) #DA for LSTM
        y=Dense(num_classes1,activation='softmax')(x)

        model=Model(inputs=inputs, outputs=y)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        history = model.fit(x2_train, y2_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=0,
                            validation_split=0.1)
        
        score = model.evaluate(x2_test, y2_test,
                               batch_size=batch_size, verbose=1)
        
        print(model.metrics_names)
        print(score)
        SINGLESCORES[(topic3,topic4)]=score[1]


        #===================================== MULTI-TASK =============================
        
        print("Learning: %s+%s VERSUS %s+%s"%(reutersTopics[topic1], reutersTopics[topic2], reutersTopics[topic3],reutersTopics[topic4]))
        

        print('Building model...')


        inputsA=Input(shape=(max_length,))
        x1=Embedding(300000, 16)(inputsA)
        x1=Dense(64,activation='relu')(x1)
        x1=Flatten()(x1) #DA for LSTM

        inputsB=Input(shape=(max_length,))
        x2=Embedding(300000, 16)(inputsB)
        x2=Dense(64,activation='relu')(x2)
        x2=Flatten()(x2) #DA for LSTM

        y1=Dense(num_classes1,activation='softmax')(x1)
        y2=Dense(num_classes2,activation='softmax')(x2)

        model=Model(inputs=[inputsA, inputsB],outputs=[y1,y2])


        def custom_loss(a,b):
            def loss(y_true,y_pred):
                e1=keras.losses.categorical_crossentropy(y_true,y_pred)
                e2=keras.losses.mean_squared_error(a,b)
                e3=1.0-keras.losses.cosine_proximity(a,b)
                e4=K.mean(K.square(a-b), axis=-1)
                return e1+e2+e3+e4
            return loss
    
    
        model.compile(loss=custom_loss(x1,x2),
                  optimizer='adam',
                  metrics=['accuracy'])

        history = model.fit([x1_train,x2_train],[y1_train,y2_train],
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1)
    
        score = model.evaluate([x1_test,x2_test],[y1_test,y2_test],
                           batch_size=batch_size, verbose=1)


        print(model.metrics_names)
        print(score)

        if score[3]>SINGLESCORES[(topic1,topic2)]:
            print("TOPICS %s+%s improved with %s+%s: %f => %f"%(reutersTopics[topic1],reutersTopics[topic2],reutersTopics[topic3],reutersTopics[topic4],SINGLESCORES[(topic1,topic2)],score[3]))
            SUCCES["TOPICS %s+%s improved with %s+%s: %f => %f"%(reutersTopics[topic1],reutersTopics[topic2],reutersTopics[topic3],reutersTopics[topic4],SINGLESCORES[(topic1,topic2)],score[3])]=1

        if score[4]>SINGLESCORES[(topic3,topic4)]:
            print("TOPICS %s+%s improved with %s+%s: %f => %f"%(reutersTopics[topic3],reutersTopics[topic4],reutersTopics[topic1],reutersTopics[topic2],SINGLESCORES[(topic3,topic4)],score[4]))
            SUCCES["TOPICS %s+%s improved with %s+%s: %f => %f"%(reutersTopics[topic3],reutersTopics[topic4],reutersTopics[topic1],reutersTopics[topic2],SINGLESCORES[(topic3,topic4)],score[4])]=1
            

        MULTISCORES[(topic1,topic2)]=(score[3],(topic3,topic4))
        MULTISCORES[(topic3,topic4)]=(score[4],(topic1,topic2))
                      
        print("==============================================")

f=open("success-soft.txt","w")
for succes in SUCCES:
    f.write(succes+"\n")
f.close()
exit(0)



