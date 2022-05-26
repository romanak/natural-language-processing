from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import merge,concatenate,recurrent, LSTM, Dense, Merge, Reshape
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
import numpy as np

import sys
import re
from keras.utils.vis_utils import plot_model



def create_tokenizer(trainingdata, testdata):
    f=open(trainingdata, "r")
    text=[]

    max_story_len=0
    max_query_len=0
    
    for line in f:
        m=re.match("^\d+\s([^\.]+)[\.].*",line.rstrip())
        if m:
            text.append(m.group(1))
            max_story_len=max(max_story_len,len(m.group(1).split(" ")))
        else:
            m=re.match("^\d+\s([^\?]+)[\?]\s\t([^\t]+)",line.rstrip())
            if m:
                text.append(m.group(1)+' '+m.group(2))
                max_query_len=max(max_query_len,len(m.group(1).split(" ")))                                
    f.close()
    
    f=open(testdata, "r")
    for line in f:
        m=re.match("^\d+\s([^\.]+)[\.].*",line.rstrip())
        if m:
            text.append(m.group(1))
            max_story_len=max(max_story_len,len(m.group(1).split(" ")))
        else:
            m=re.match("^\d+\s([^\?]+)[\?].*",line.rstrip())
            if m:
                text.append(m.group(1))
                max_query_len=max(max_query_len,len(m.group(1).split(" ")))                                
    f.close()
    
    vocabulary=set([word for word in text])
    max_words = len(vocabulary)
    tokenizer = Tokenizer(num_words=max_words, char_level=False, split=' ')
    tokenizer.fit_on_texts(text)
    return tokenizer,len(vocabulary),max_story_len, max_query_len


def vectorize(s, tokenizer):
    vector=tokenizer.texts_to_sequences([s])
    return vector[0]


    
def process_stories(filename,tokenizer,max_story_len,max_query_len,vocab_size,use_context=False):
    f=open(filename,"r")
    X=[]
    Q=[]
    y=[]
    
    for line in f:
        m=re.match("^(\d+)\s(.+)\.",line.rstrip())
        if m:
            if int(m.group(1))==1:
                story={}
            story[int(m.group(1))]=m.group(2)
        else:
            m=re.match("^\d+\s(.+)\?\s\t([^\t]+)\t(.+)",line.rstrip())
            if m:
                question=m.group(1)
                answer=m.group(2)
                answer_ids=[int(x) for x in m.group(3).split(" ")]
                if use_context==False:
                    facts=' '.join([story[id] for id in answer_ids])
                    vectorized_fact=vectorize(facts,tokenizer)
                else:
                    vectorized_fact=vectorize(' '.join(story.values()),tokenizer)
                vectorized_question=vectorize(question,tokenizer)
                vectorized_answer=vectorize(answer,tokenizer)
                #vector=np.append(vectorized_fact, vectorized_question)
                #X.append(vector)
                X.append(vectorized_fact)
                Q.append(vectorized_question)
                answer=np.zeros(vocab_size)
                answer[vectorized_answer[0]]=1
                y.append(answer)
    f.close()

    X=pad_sequences(X,maxlen=max_story_len)
    Q=pad_sequences(Q,maxlen=max_query_len)
    
    return np.array(X),np.array(Q),np.array(y)




def process_stories_n_context(filename,tokenizer,max_story_len,max_query_len,vocab_size,use_context=0):
    f=open(filename,"r")
    X=[]
    Q=[]
    y=[]
    for line in f:
        m=re.match("^(\d+)\s(.+)\.",line.rstrip())
        if m:
            if int(m.group(1))==1:
                story={}
            story[int(m.group(1))]=m.group(2)
        else:
            m=re.match("^\d+\s(.+)\?\s\t([^\t]+)\t(.+)",line.rstrip())
            if m:
                question=m.group(1)
                answer=m.group(2)
                answer_ids=[int(x) for x in m.group(3).split(" ")]
                if use_context==0:
                    facts=' '.join([story[id] for id in answer_ids])
                    vectorized_fact=vectorize(facts,tokenizer)
                else:
                    x=min(use_context, len(story))
                    facts=' '.join([story[id] for id in answer_ids])+' '
                    n=0
                    for id in story:
                        if n<x and id not in answer_ids:
                            facts+=story[id]+' '
                            n+=1                    
                    vectorized_fact=vectorize(facts,tokenizer)
                vectorized_question=vectorize(question,tokenizer)
                vectorized_answer=vectorize(answer,tokenizer)
                X.append(vectorized_fact)
                Q.append(vectorized_question)
                answer=np.zeros(vocab_size)
                answer[vectorized_answer[0]]=1
                y.append(answer)
    f.close()

    X=pad_sequences(X,maxlen=max_story_len)
    Q=pad_sequences(Q,maxlen=max_query_len)
    
    return np.array(X),np.array(Q),np.array(y)



def process_stories_n_context_stateful(filename,tokenizer,max_story_len,max_query_len,vocab_size,use_context=0):
    f=open(filename,"r")
    X=[]
    Q=[]
    y=[]
    n=0
    Stories=[]
    for line in f:
        m=re.match("^(\d+)\s(.+)\.",line.rstrip())
        if m:
            if int(m.group(1))==1:
                story={}
            story[int(m.group(1))]=m.group(2)
        else:
            m=re.match("^\d+\s(.+)\?\s\t([^\t]+)\t(.+)",line.rstrip())
            if m:
                question=m.group(1)
                vectorized_question=vectorize(question,tokenizer)
                answer=m.group(2)
                vectorized_answer=vectorize(answer,tokenizer)
                label=np.zeros(vocab_size)
                label[vectorized_answer[0]]=1
                Stories.append([])

                for id in story:
                    fact=story[id]
                    vectorized_fact=vectorize(fact, tokenizer)
                    Stories[n].append((vectorized_fact, vectorized_question))
                    y.append(label)
                n+=1

    f.close()

    m=max(max_story_len, max_query_len)

    for i in range(len(Stories)):
        padded=[]
        for fact_question_pair in Stories[i]:
            X=fact_question_pair[0]
            Q=fact_question_pair[1]
            X_pad=pad_sequences([X],maxlen=m)
            Q_pad=pad_sequences([Q],maxlen=m)
            padded.append((X_pad[0],Q_pad[0]))
        Stories[i]=padded

    
    #return np.array(X),np.array(Q),np.array(y)
    return Stories,np.array(y)




# Several options:
# 2 embeddings, concat
# 2 embeddings, each feeding into RNN, concat
# 2 embeddings, each feeding into RNN, apply RNN to result
# Similar for other types: LSTM, GRU, memn2n


def create_model(trainingData, testData, context):

    tokenizer,vocab_size, max_story_len, max_query_len=create_tokenizer(trainingData,testData)
    
#    X_tr,Q_tr,y_tr=process_stories(trainingData,tokenizer,max_story_len, max_query_len,vocab_size,use_context=context)
#    X_te,Q_te,y_te=process_stories(testData,tokenizer,max_story_len, max_query_len,vocab_size,use_context=context)

    X_tr,Q_tr,y_tr=process_stories_n_context(trainingData,tokenizer,max_story_len, max_query_len,vocab_size,use_context=context)
    X_te,Q_te,y_te=process_stories_n_context(testData,tokenizer,max_story_len, max_query_len,vocab_size,use_context=context)

    embedding=layers.Embedding(vocab_size,100)
    
    story = layers.Input(shape=(max_story_len,), dtype='int32')
    encoded_story = embedding(story)
    encoded_story = LSTM(30)(encoded_story)

    question = layers.Input(shape=(max_query_len,), dtype='int32')
    encoded_question = embedding(question)
    encoded_question = LSTM(30)(encoded_question)

    merged = layers.concatenate([encoded_story, encoded_question])
    
    preds = layers.Dense(vocab_size, activation='softmax')(merged)

    model = Model([story, question], preds)
    model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    print vocab_size
    print max_story_len, max_query_len
    model.summary()
    
    return X_tr,Q_tr,y_tr,X_te,Q_te,y_te,max_story_len,model



def create_model_stateful_old(trainingData, testData, context):

    tokenizer,vocab_size, max_story_len, max_query_len=create_tokenizer(trainingData,testData)

    m=max(max_story_len, max_query_len)
    
    X_tr,Q_tr,y_tr=process_stories_n_context_stateful(trainingData,tokenizer,max_story_len, max_query_len,vocab_size,use_context=context)
    X_te,Q_te,y_te=process_stories_n_context_stateful(testData,tokenizer,max_story_len, max_query_len,vocab_size,use_context=context)

    embedding=layers.Embedding(vocab_size,100)
    
    story_question = layers.Input(shape=(m*2,),batch_shape=(1,12), dtype='int32')
    encoded_story_question = embedding(story_question)

#    encoded_story_question=encoded_story_question.reshape((1,100,1))
    
    merged=LSTM(30,input_shape=(1,12,1),return_sequences=False, stateful=True)(encoded_story_question)

#    merged=LSTM(30, input_shape=(100,1))(encoded_story_question)
    
    preds = layers.Dense(vocab_size, activation='softmax')(merged)

    model = Model(story_question, preds)
    
    model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    print vocab_size
    print max_story_len, max_query_len
    model.summary()
    
    return X_tr,Q_tr,y_tr,X_te,Q_te,y_te,max_story_len,model


def create_model_stateful(trainingData, testData, context):

    tokenizer,vocab_size, max_story_len, max_query_len=create_tokenizer(trainingData,testData)

    m=max(max_story_len, max_query_len)
    
    Stories_tr,y_tr=process_stories_n_context_stateful(trainingData,tokenizer,max_story_len, max_query_len,vocab_size,use_context=context)
    Stories_te,y_te=process_stories_n_context_stateful(testData,tokenizer,max_story_len, max_query_len,vocab_size,use_context=context)

    embedding=layers.Embedding(vocab_size,20) # 100
    
    story_question = layers.Input(shape=(m*2,),batch_shape=(1,12), dtype='int32')
    encoded_story_question = embedding(story_question)

#    encoded_story_question=encoded_story_question.reshape((1,100,1))
    
    merged=LSTM(30,input_shape=(1,12,1),return_sequences=False, stateful=True)(encoded_story_question)

#    merged=LSTM(30, input_shape=(100,1))(encoded_story_question)
    
    preds = layers.Dense(vocab_size, activation='softmax')(merged)

    model = Model(story_question, preds)
    
    model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    print vocab_size
    print max_story_len, max_query_len
    model.summary()
    
    return Stories_tr,y_tr,Stories_te,y_te,max_story_len,model



def run_evaluate(trainingData, testData, context):

    X_tr,Q_tr,y_tr,X_te,Q_te,y_te,max_story_len,model=create_model(trainingData,testData,context)

    model.fit([X_tr, Q_tr], y_tr,
              batch_size=32,
              epochs=10,
              verbose=1,
              validation_split=0.1)

    print('Evaluation')
    loss, acc = model.evaluate([X_te,Q_te], y_te,
                               batch_size=32)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


def run_evaluate_stateful_old(trainingData, testData, context):
    X_tr,Q_tr,y_tr,X_te,Q_te,y_te,max_story_len,model=create_model_stateful(trainingData,testData,context)

    max_epoch=10
    for epoch in range(1,max_epoch):
        print "Epoch %d/%d"%(epoch,max_epoch)
        
        mean_tr_acc = []
        mean_tr_loss = []

        for i in range(len(X_tr)):
            #print "%d/%d"%(i,len(X_tr))
            X=np.array(X_tr[i])
            Q=np.array(Q_tr[i])
            XQ=np.append(X,Q)
            XQ=XQ.reshape((1,12))
            y=np.array(y_tr[i])
            #index=np.where(y_tr[i]==1)[0][0]
            #if index not in XQ[0]:
            #    y=np.zeros(len(y_tr[i]))
            y=y.reshape((1,148))
            tr_loss, tr_acc = model.train_on_batch(XQ,y)
            mean_tr_acc.append(tr_acc)
            mean_tr_loss.append(tr_loss)
        model.reset_states()

        print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
        print('loss training = {}'.format(np.mean(mean_tr_loss)))
        print('____________________________________')
        
        mean_te_acc = []
        mean_te_loss = []
        for i in range(len(X_te)):
            X=np.array(X_te[i])
            Q=np.array(Q_te[i])
            XQ=np.append(X,Q)
            XQ=XQ.reshape((1,12))
            y=np.array(y_te[i])
            #index=np.where(y_te[i]==1)[0][0]
            #if index not in XQ[0]:
            #    y=np.zeros(len(y_te[i]))
            y=y.reshape((1,148))
            te_loss, te_acc = model.test_on_batch(XQ, y)
            mean_te_acc.append(te_acc)
            mean_te_loss.append(te_loss)
        model.reset_states()

        print('accuracy testing = {}'.format(np.mean(mean_te_acc)))
        print('loss testing = {}'.format(np.mean(mean_te_loss)))
        print('___________________________________')


def run_evaluate_stateful(trainingData, testData, context):
    Stories_tr,y_tr,Stories_te,y_te,max_story_len,model=create_model_stateful(trainingData,testData,context)

    max_epoch=10
    for epoch in range(1,max_epoch):
        print "Epoch %d/%d"%(epoch,max_epoch)
        
        mean_tr_acc = []
        mean_tr_loss = []


        for i in range(len(Stories_tr)):
            for j in range(len(Stories_tr[i])):
                X=np.array(Stories_tr[i][j][0])
                Q=np.array(Stories_tr[i][j][1])
                XQ=np.append(X,Q)
                XQ=XQ.reshape((1,12))
                y=np.array(y_tr[i])
                index=np.where(y_tr[i]==1)[0][0]
                print index
                #if index not in XQ[0]:
                #    y=np.zeros(len(y_tr[i]))
                y=y.reshape((1,148))
                tr_loss, tr_acc = model.train_on_batch(XQ,y)
                mean_tr_acc.append(tr_acc)
                mean_tr_loss.append(tr_loss)
        model.reset_states()

        print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
        print('loss training = {}'.format(np.mean(mean_tr_loss)))
        print('____________________________________')
        
        mean_te_acc = []
        mean_te_loss = []


        for i in range(len(Stories_te)):
            for j in range(len(Stories_te[i])):
                X=np.array(Stories_te[i][j][0])
                Q=np.array(Stories_te[i][j][1])
                XQ=np.append(X,Q)
                XQ=XQ.reshape((1,12))
                y=np.array(y_te[i])
                #index=np.where(y_te[i]==1)[0][0]
                #if index not in XQ[0]:
                #    y=np.zeros(len(y_te[i]))
                y=y.reshape((1,148))
                te_loss, te_acc = model.test_on_batch(XQ,y)
                mean_te_acc.append(te_acc)
                mean_te_loss.append(te_loss)
        model.reset_states()

        
        print('accuracy testing = {}'.format(np.mean(mean_te_acc)))
        print('loss testing = {}'.format(np.mean(mean_te_loss)))
        print('___________________________________')

        
if __name__=="__main__":        
    run_evaluate_stateful(sys.argv[1],sys.argv[2],eval(sys.argv[3]))
#    run_evaluate(sys.argv[1],sys.argv[2],eval(sys.argv[3]))
    print "DONE"
