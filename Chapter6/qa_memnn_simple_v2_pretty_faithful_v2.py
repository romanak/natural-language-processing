from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent, LSTM, Bidirectional, Input, Dropout,add, dot, concatenate, Activation, Permute, Dense,merge,Reshape,Flatten, Lambda, Dot
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
import numpy as np
import keras.backend 
import sys
import re
from keras.utils.vis_utils import plot_model

from keras import backend as K


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

def vectorize_label(l):
    if l=='N':
        return np.array([0])
    else:
        return np.array([1])

    
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
                facts=[story[id] for id in answer_ids]

                vectorized_facts=[]
                for fact in facts:
                    vectorized_facts.append(np.array(vectorize(fact,tokenizer)))

                l=len(vectorized_facts)
                if l>max_story_len:
                    max_story_len=l
                vectorized_question=vectorize(question,tokenizer)
                l=len(vectorized_question)
                if l>max_query_len:
                    max_query_len=l

                vectorized_answer=vectorize_label(answer)
                
                X.append(vectorized_facts)
                Q.append(vectorized_question)
                answer=np.zeros(2)
                answer[vectorized_answer[0]]=1
                y.append(answer)
    f.close()

    return np.array(X),np.array(Q),np.array(y), max_story_len, max_query_len




def pad_data(X,Q,max_story_len, max_query_len):
    X_new=[]
    Q_new=[]
    m=max(max_story_len, max_query_len)

    for x in X:
        X_new.append(pad_sequences(x,maxlen=m))

    #X_new=pad_sequences(X_new,maxlen=m)
   # Q_new=pad_sequences(Q,maxlen=m)
   
    for q in Q:
        Q_new.append(pad_sequences([q],maxlen=m))

    return np.array(X_new),np.array(Q_new)


# Several options:
# 2 embeddings, concat
# 2 embeddings, each feeding into RNN, concat
# 2 embeddings, each feeding into RNN, apply RNN to result
# Similar for other types: LSTM, GRU, memn2n



# Lambda functions
#-------------------

def lambda_multiply(x,n):
    x_prime = tf.reshape(x, (-1, n, 1))
    x_transpose = tf.transpose(x_prime, perm=[0,2, 1])
    return tf.batch_matmul(x_transpose,x_prime)




def create_model(trainingData, testData, context):

#    tokenizer,vocab_size, max_story_len, max_query_len=create_tokenizer(trainingData,testData)

    tokenizer,vocab_size=create_tokenizer(trainingData,testData)

    X_tr,Q_tr,y_tr,max_story_len_tr, max_query_len_tr=process_stories_n_context(trainingData,tokenizer,vocab_size,use_context=context)
    X_te,Q_te,y_te, max_story_len_te, max_query_len_te=process_stories_n_context(testData,tokenizer,vocab_size,use_context=context)
    
    max_story_len=max(max_story_len_tr, max_story_len_te)
    max_query_len=max(max_query_len_tr, max_query_len_te)

    print "Max story:", max_story_len

    print "Padding data"

    X_tr, Q_tr=pad_data(X_tr,Q_tr,max_story_len, max_query_len)
    X_te, Q_te=pad_data(X_te,Q_te,max_story_len, max_query_len)

    print "Data padded"

    print max_story_len, max_query_len
    print "-----------------------"
    print X_tr[0], Q_tr[0]
    print X_tr.shape, Q_tr.shape

    
    m=max(max_story_len, max_query_len)  

    input_facts = Input((2,m,))

    question = Input((1,m,))

    # A
    A= Embedding(input_dim=vocab_size,
                 output_dim=m)
    # C

    C=Embedding(input_dim=vocab_size,
                output_dim=max_query_len)
    # B
    B=Embedding(input_dim=vocab_size,
                output_dim=m,
                input_length=max_query_len)

    def embed_A(x):
        return A(x[0])

    def embed_C(x):
        return C(x[0])
    
    #input_A=Lambda(embed_A, output_shape =(max_story_len, m, 64))([input_facts])

    input_A=Lambda(embed_A)([input_facts])
    print "input_A", keras.backend.int_shape(input_A)
    print "question", keras.backend.int_shape(question)
    input_C=Lambda(embed_C)([input_facts])
    print "input_C", keras.backend.int_shape(input_C)
    question_B = B(question)
    print "question_B", keras.backend.int_shape(question_B)   

    input_question_match = dot([input_A,question_B],axes=(2,2))

    print "OK!!!!!!!!!!!!!"
    print "IQ",keras.backend.int_shape(input_question_match)
    Probs = Activation('softmax')(input_question_match)
    print "Probs",keras.backend.int_shape(Probs)
    O = add([Probs, input_C])  # (samples, max_story_len, max_query_len)
    print "O",keras.backend.int_shape(O)

    print "Q_B",keras.backend.int_shape(question_B)
    O=Flatten()(O)
    question_B=Flatten()(question_B)
    final_match = concatenate([O,question_B])
    print "FM",keras.backend.int_shape(final_match)
    size=keras.backend.int_shape(final_match)[1]
    print "FM size",size

    weights =  Dense(size, activation='softmax')(final_match)
    merged=merge([final_match, weights], mode='mul')
    answer = Dense(2)(merged)  # (samples, vocab_size)
    
    #answer = Dense(2)(O)  # (samples, vocab_size)

    answer = Activation('softmax')(answer)

    
    print keras.backend.int_shape(answer)
    print "Done"


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
        
