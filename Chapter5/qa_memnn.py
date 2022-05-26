from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent, LSTM, Bidirectional, Input, Dropout,add, dot, concatenate, Activation, Permute, Dense
from keras.models import Model, Sequential
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

    input_sequence = Input((max_story_len,))
    question = Input((max_query_len,))

    # A
    input_encoder_m = Sequential()
    input_encoder_m.add(Embedding(input_dim=vocab_size,
                              output_dim=64))
    #input_encoder_m.add(Dropout(0.3))

    # C
    input_encoder_c = Sequential()
    input_encoder_c.add(Embedding(input_dim=vocab_size,
                              output_dim=max_query_len))
    #input_encoder_c.add(Dropout(0.3))


    # B
    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=max_query_len))
    #question_encoder.add(Dropout(0.3))

    input_encoded_m = input_encoder_m(input_sequence)
    input_encoded_c = input_encoder_c(input_sequence)
    question_encoded = question_encoder(question)

    # compute a 'match' between the first input vector sequence
    # and the question vector sequence
    # shape: `(samples, max_story_len, max_query_len)`
    match = dot([input_encoded_m, question_encoded], axes=(2, 2))
    match = Activation('softmax')(match)

    # add the match matrix with the second input vector sequence
    response = add([match, input_encoded_c])  # (samples, max_story_len, max_query_len)
    response = Permute((2, 1))(response)  # (samples, max_query_len, max_story_len)

    # concatenate the match matrix with the question vector sequence
    answer = concatenate([response, question_encoded])

    # the original paper uses a matrix multiplication for this reduction step.
    # we choose to use a RNN instead.
    answer = LSTM(32)(answer)  # (samples, 32)

    # one regularization layer -- more would probably be needed.
    answer = Dropout(0.3)(answer)
    answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
    # we output a probability distribution over the vocabulary
    answer = Activation('softmax')(answer)

    # build the final model
    model = Model([input_sequence, question], answer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

    # train
    model.fit([X, Q], y,
              batch_size=32,
              epochs=120)
#              validation_data=([X_te, Q_te], y_te))

    print('Evaluation')
    loss, acc = model.evaluate([X_te,Q_te], y_te,
                               batch_size=32)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
