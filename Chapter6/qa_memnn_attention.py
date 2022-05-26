from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent, LSTM, Bidirectional, Input, Dropout,add, dot, concatenate, Activation, Permute, Dense,merge,Reshape,Flatten
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
import numpy as np
import keras.backend 
import sys
import re
from keras.utils.vis_utils import plot_model


def create_tokenizer(trainingdata, testdata):
    f=open(trainingdata, "r")
    text=[]

    for line in f:
        m=re.match("^\d+\s([^\.]+)[\.].*",line.rstrip())
        if m:
            text.append(m.group(1))
        else:
            m=re.match("^\d+\s([^\?]+)[\?]\s\t([^\t]+)",line.rstrip())
            if m:
                text.append(m.group(1)+' '+m.group(2))
    f.close()
    
    f=open(testdata, "r")
    for line in f:
        m=re.match("^\d+\s([^\.]+)[\.].*",line.rstrip())
        if m:
            text.append(m.group(1))
        else:
            m=re.match("^\d+\s([^\?]+)[\?].*",line.rstrip())
            if m:
                text.append(m.group(1))
    f.close()
    
    vocabulary=set([word for word in text])
    max_words = len(vocabulary)
    tokenizer = Tokenizer(num_words=max_words, char_level=False, split=' ')
    tokenizer.fit_on_texts(text)
    return tokenizer, max_words


def vectorize(s, tokenizer):
    vector=tokenizer.texts_to_sequences([s])
    return vector[0]


    
def process_stories(filename,tokenizer,max_story_len,max_query_len,vocab_size,use_context=False):
    f=open(filename,"r")
    X=[]
    Q=[]
    y=[]
    n_questions=0
    story={}
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
    story={}
    for line in f:
#        print line
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
                all_facts=' '.join([story[id] for id in story])
                facts_v=vectorize(facts,tokenizer)
                all_facts_v=vectorize(all_facts,tokenizer)


                if use_context==0:                    
                    vectorized_fact=facts_v
                elif use_context==-1:
                    vectorized_fact=all_facts_v
                else:
                    x=min(use_context, len(story))
                    facts=' '.join([story[id] for id in answer_ids])+' '                   
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
                answer=np.zeros(vocab_size)
                answer[vectorized_answer[0]]=1
                y.append(answer)
    f.close()

    return np.array(X),np.array(Q),np.array(y), max_story_len, max_query_len




def pad_data(X,Q,max_story_len, max_query_len):
    X=pad_sequences(X,maxlen=max_story_len)
    Q=pad_sequences(Q,maxlen=max_query_len)
    return np.array(X),np.array(Q)


# Several options:
# 2 embeddings, concat
# 2 embeddings, each feeding into RNN, concat
# 2 embeddings, each feeding into RNN, apply RNN to result
# Similar for other types: LSTM, GRU, memn2n



def create_model(trainingData, testData, context):

#    tokenizer,vocab_size, max_story_len, max_query_len=create_tokenizer(trainingData,testData)

    tokenizer,vocab_size=create_tokenizer(trainingData,testData)

    X_tr,Q_tr,y_tr,max_story_len_tr, max_query_len_tr=process_stories_n_context(trainingData,tokenizer,vocab_size,use_context=context)
    X_te,Q_te,y_te, max_story_len_te, max_query_len_te=process_stories_n_context(testData,tokenizer,vocab_size,use_context=context)
  
    max_story_len=max(max_story_len_tr, max_story_len_te)
    max_query_len=max(max_query_len_tr, max_query_len_te)

    print "Max story:", max_story_len
    
    X_tr, Q_tr=pad_data(X_tr,Q_tr,max_story_len, max_query_len)
    X_te, Q_te=pad_data(X_te,Q_te,max_story_len, max_query_len)
    
    input_facts = Input((max_story_len,))
    question = Input((max_query_len,))

    # A
    A= Embedding(input_dim=vocab_size,
                              output_dim=64)
    # C
    C=Embedding(input_dim=vocab_size,
                              output_dim=max_query_len)
    # B
    B=Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=max_query_len)

    input_A = A(input_facts)
    input_C = C(input_facts)
    question_B = B(question)

    print keras.backend.int_shape(input_A),keras.backend.int_shape(input_C),keras.backend.int_shape(question_B)
    
    input_question_match = dot([input_A, question_B], axes=(2, 2))


    attention_probs = Dense(5, activation='softmax', name='attention_probs')(input_question_match)
    attention_mul = merge([input_question_match, attention_probs], output_shape=5, name='attention_mul', mode='mul')


    
    Probs = Activation('softmax')(attention_mul) #input_question_match)
    print keras.backend.int_shape(Probs)

    print "IQM",keras.backend.int_shape(input_question_match),keras.backend.int_shape(Probs),keras.backend.int_shape(input_A),keras.backend.int_shape(question_B)

    O = add([Probs, input_C])  # (samples, max_story_len, max_query_len)
    O = Permute((2, 1))(O)  # (samples, max_query_len, max_story_len)

    print keras.backend.int_shape(O),keras.backend.int_shape(question_B)
    final_match = concatenate([O, question_B])

    size=keras.backend.int_shape(final_match)[2]
    weights = Dense(size, activation='softmax')(final_match)
    merged=merge([final_match, weights], mode='mul')
    answer=Flatten()(merged)
    
    answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
    answer = Activation('softmax')(answer)

    model = Model([input_facts, question], answer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

    model.summary()
    
    return X_tr,Q_tr,y_tr,X_te,Q_te,y_te,model



def run_evaluate(trainingData, testData, context):

    X_tr,Q_tr,y_tr,X_te,Q_te,y_te,model=create_model(trainingData,testData,context)

    model.fit([X_tr, Q_tr], y_tr,
              batch_size=32,
              epochs=100)
    print('Evaluation')
    loss, acc = model.evaluate([X_te,Q_te], y_te,
                               batch_size=32)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))



if __name__=="__main__":        
        run_evaluate(sys.argv[1],sys.argv[2],eval(sys.argv[3]))
        
