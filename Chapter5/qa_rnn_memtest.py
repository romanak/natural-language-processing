from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent, SimpleRNN, Dense
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
import numpy as np

import sys
import re
from keras.utils.vis_utils import plot_model

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


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



def process_stories_n_context(filename,tokenizer,vocab_size,use_context=0):
    f=open(filename,"r")
    X=[]
    Q=[]
    y=[]
    max_story_len=0
    max_query_len=0
    
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

                facts=' '.join([story[id] for id in answer_ids])
                facts_v=vectorize(facts,tokenizer)

                if use_context==0:
                    vectorized_fact=facts_v
                else:
                    x=min(use_context, len(story))
                    facts=' '.join([story[id] for id in answer_ids])+' '
                    answer=facts
                    n=0
                    for id in story:
                        if n<x and id not in answer_ids:
                            facts+=story[id]+' '
                            n+=1                    
                    vectorized_fact=vectorize(facts,tokenizer)
                l=len(vectorized_fact)
                if l>max_story_len:
                    max_story_len=l
                    
                vectorized_question=vectorize(question,tokenizer)
                l=len(vectorized_question)
                if l>max_query_len:
                    max_query_len=l
                    
                vectorized_answer=vectorize(answer,tokenizer)

                
                X.append(vectorized_fact)
                Q.append(vectorized_question)
                #answer=np.zeros(vocab_size)
                #answer[vectorized_answer[0]]=1
                #y.append(answer)
                y.append(vectorized_fact)
    f.close()

    return np.array(X),np.array(Q),np.array(y), max_story_len, max_query_len



def pad_data(X,Q,y,max_story_len, max_query_len):
    print "MAX:",max_story_len,max_query_len
    X=pad_sequences(X,maxlen=max_story_len)
    Q=pad_sequences(Q,maxlen=max_query_len)
    y=pad_sequences(y,maxlen=max_story_len)
    return np.array(X),np.array(Q),np.array(y)




# Several options:
# 2 embeddings, concat
# 2 embeddings, each feeding into RNN, concat
# 2 embeddings, each feeding into RNN, apply RNN to result
# Similar for other types: LSTM, GRU, memn2n


def create_model(trainingData, testData, context):

    tokenizer,vocab_size, max_story_len, max_query_len=create_tokenizer(trainingData,testData)
    
#    X_tr,Q_tr,y_tr=process_stories(trainingData,tokenizer,max_story_len, max_query_len,vocab_size,use_context=context)
#    X_te,Q_te,y_te=process_stories(testData,tokenizer,max_story_len, max_query_len,vocab_size,use_context=context)

    X_tr,Q_tr,y_tr,max_story_len_tr, max_query_len_tr=process_stories_n_context(trainingData,tokenizer,vocab_size,use_context=context)
    X_te,Q_te,y_te, max_story_len_te, max_query_len_te=process_stories_n_context(testData,tokenizer,vocab_size,use_context=context)

    max_story_len=max(max_story_len_tr, max_story_len_te)
    max_query_len=max(max_query_len_tr, max_query_len_te)

    X_tr, Q_tr,y_tr=pad_data(X_tr,Q_tr, y_tr,max_story_len, max_query_len)
    X_te, Q_te,y_te=pad_data(X_te,Q_te, y_te,max_story_len, max_query_len)
    
    embedding=layers.Embedding(vocab_size,100)
    
    story = layers.Input(shape=(max_story_len,), dtype='int32')
    encoded_story = embedding(story)
    encoded_story = SimpleRNN(300)(encoded_story)

#    question = layers.Input(shape=(max_query_len,), dtype='int32')
 #   encoded_question = embedding(question)
  #  encoded_question = LSTM(30)(encoded_question)

   # merged = layers.concatenate([encoded_story, encoded_question])

    merged=encoded_story
    preds = layers.Dense(max_story_len, activation='relu')(merged)
    preds = layers.Dense(max_story_len, activation='relu')(merged)

#    model = Model([story, question], preds)
    model = Model([story], preds)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', #'mean_squared_error',
                  metrics=['mae'])


    model.summary()

    
    return X_tr,Q_tr,y_tr,X_te,Q_te,y_te,model



def run_evaluate(trainingData, testData, context):

    X_tr,Q_tr,y_tr,X_te,Q_te,y_te,model=create_model(trainingData,testData,context)

    model.fit([X_tr], y_tr,
              batch_size=32,
              epochs=10,
              verbose=1,
              validation_split=0.1)

    print('Evaluation')

    preds=model.predict([X_te])


    mse = mean_squared_error(y_te, preds)
    mae = mean_absolute_error(y_te, preds)
    print('Mae / Mse = {:.4f} / {:.4f}'.format(mae,mse))


if __name__=="__main__":        
        run_evaluate(sys.argv[1],sys.argv[2],eval(sys.argv[3]))
