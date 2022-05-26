from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent, LSTM, Bidirectional
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
    n_questions=0
    
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


# Several options:
# 2 embeddings, concat
# 2 embeddings, each feeding into RNN, concat
# 2 embeddings, each feeding into RNN, apply RNN to result
# Similar for other types: LSTM, GRU, memn2n

if __name__=="__main__":

    tokenizer,vocab_size, max_story_len, max_query_len=create_tokenizer(sys.argv[1],sys.argv[2])
    
    X,Q,y=process_stories(sys.argv[1],tokenizer,max_story_len, max_query_len,vocab_size,use_context=True)

    X_te,Q_te,y_te=process_stories(sys.argv[2],tokenizer,max_story_len, max_query_len,vocab_size,use_context=True)

    emb=layers.Embedding(vocab_size,100)
    
    sentence = layers.Input(shape=(max_story_len,), dtype='int32')
#    encoded_sentence = layers.Embedding(vocab_size, 100)(sentence)
    encoded_sentence = emb(sentence)
    #encoded_sentence = Bidirectional(LSTM(100))(encoded_sentence)

    encoded_sentence = LSTM(100, return_sequences=True)(encoded_sentence)
    encoded_sentence = LSTM(100, return_sequences=True)(encoded_sentence)
    
    encoded_sentence = LSTM(100)(encoded_sentence)

    question = layers.Input(shape=(max_query_len,), dtype='int32')
    #encoded_question = layers.Embedding(vocab_size, 100)(question)
    encoded_question = emb(question)
    #encoded_question = Bidirectional(LSTM(100))(encoded_question)

    encoded_question = LSTM(100, return_sequences=True)(encoded_question)
    encoded_question = LSTM(100, return_sequences=True)(encoded_question)
    
    encoded_question = LSTM(100)(encoded_question)

    merged = layers.concatenate([encoded_sentence, encoded_question])
    preds = layers.Dense(vocab_size, activation='softmax')(merged)

    model = Model([sentence, question], preds)
    model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    print('Training')
    model.fit([X, Q], y,
              batch_size=32,
              epochs=10,
              verbose=1,
              validation_split=0.1)

    print('Evaluation')
    loss, acc = model.evaluate([X_te,Q_te], y_te,
                               batch_size=32)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
