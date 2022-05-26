import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import re
import numpy as np


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


if __name__=="__main__":
    f=open(sys.argv[1],"r")
    M={}
    for line in f:
        if re.match("^\d+\s+\d+$",line.rstrip()):
            continue # header
        if re.match("^<unk>\s.+$",line.rstrip()):
            continue # header
        m=re.match("^([^\s]+)\s(.+)$",line.rstrip())
        if m:
            M[m.group(1)]=np.array(m.group(2).split(" "))

    f.close()
    print M["bad"]
    tsne_plot(M,100)
