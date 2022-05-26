from __future__ import absolute_import
from keras import backend as K
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense, merge, Reshape,Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.objectives import mse
import global_settings as G
from sentences_generator import Sentences
import vocab_generator as V_gen
import save_embeddings as S
import sys
import re
from keras import optimizers

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from scipy.spatial.distance import cosine


context_size = 2*G.window_size # 5 left, 5 right

# Creating a sentence generator from demo file
sentences = Sentences(sys.argv[1])
vocabulary = dict()
V_gen.build_vocabulary_v2(vocabulary, sentences)

#V_gen.filter_vocabulary_based_on(vocabulary, G.min_count)
#reverse_vocabulary = V_gen.generate_inverse_vocabulary_lookup(vocabulary, "vocab.txt")


# Creating CBOW model
# Model has 3 inputs
# Current word index, context words indexes and negative sampled word indexes

word_index = Input(shape=(1,))
context = Input(shape=(context_size,))
negative_samples = Input(shape=(G.negative,))


#embedding = np.random.uniform(-1.0/2.0/G.embedding_dimension, 1.0/2.0/G.embedding_dimension, (G.vocab_size+3, G.embedding_dimension))


# All the inputs are processed through a common embedding layer

vocab_size=len(vocabulary)+3
shared_embedding_layer = Embedding(input_dim=(vocab_size), output_dim=G.embedding_dimension) #,weights=[embedding])
word_embedding = shared_embedding_layer(word_index) # lookup action, produces vectors of size embedding_dimension
context_embeddings = shared_embedding_layer(context)
negative_words_embedding = shared_embedding_layer(negative_samples)

# Now the context words are averaged to get the CBOW vector
cbow = Lambda(lambda x: K.mean(x, axis=1), output_shape=(G.embedding_dimension,))(context_embeddings)

# The context is multiplied (dot product) with current word and negative sampled words
word_context_product = merge([word_embedding, cbow], mode='dot')
word_context_product = Reshape((1,))(word_context_product)
output = Dense(1, activation='sigmoid')(word_context_product) 


model = Model(input=[word_index, context], output=[output])#, negative_context_product])

# two types of results: the relation of the word and the context words, and the relation of negative words and positive words

# binary crossentropy is applied on the output
optimizer = optimizers.adam(lr=0.0001)
model.compile(optimizer='rmsprop',loss='binary_crossentropy')
print model.summary()


#model.fit_generator(V_gen.pretraining_batch_generator(sentences, vocabulary, reverse_vocabulary), samples_per_epoch=100, nb_epoch=10)

#model.fit_generator(V_gen.pretraining_batch_generator_v2(sentences, reverse_vocabulary), samples_per_epoch=32, nb_epoch=1000)


model.fit_generator(V_gen.pretraining_batch_generator_v3(sentences, vocabulary), samples_per_epoch=300, nb_epoch=10)
#model.fit_generator(V_gen.generate_batch(sentences, reverse_vocabulary,window_size), samples_per_epoch=32, nb_epoch=1000)

S.save_embeddings("embedding.txt", shared_embedding_layer.get_weights()[0], vocabulary)


def tsne_plot(model,max_words=100):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    n=0
    for word in model:
        if n<max_words: #re.match(".+_13",word):
            tokens.append(model[word])
            labels.append(word)
            n+=1
            
    
    tsne_model = TSNE(perplexity=10, n_components=2, init='pca', n_iter=1000, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(8, 8)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


def distance(x,y):
    return cosine(np.array(x,dtype=float),np.array(y,dtype=float))

def  get_knn(model,sentences):
    for s in sentences:
        words=s.split(" ")
        for word in words:
            if word =='':
                continue
            min_dist=10e4
            for w in model:
                if w == word:
                    continue
                d=distance(model[word],model[w])
                if d<min_dist:
                    min_dist=d
                    winner=w
            print word,winner,min_dist 

    
def tsne_plot_sent(model,sentences,max_words=100):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    n=0
    Seen={}
    for s in sentences:
        words=s.split(" ")
        for word in words:
            if n<max_words and word not in Seen: #re.match(".+_13",word):
                tokens.append(model[word])
                labels.append(word)
                n+=1
                Seen[word]=1
            else:
                break
            
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=10000, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(8, 8)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


f=open("embedding.txt","r")
n=0
model={}
for line in f:
   if n:
       m=re.match("^([^\t]+)\t(.+)$",line.rstrip())
       if m:
           model[m.group(1)]=np.array(m.group(2).split(" "))
   n+=1
f.close()


#tsne_plot_sent(model,sentences,200)

get_knn(model,sentences)
    
