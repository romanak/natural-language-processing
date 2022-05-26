import sys
from os import listdir
from os.path import isfile,join
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import hashing_trick
from keras.utils import np_utils
from nltk import ngrams
import numpy as np
import re

def load_embedding_zipped(f, vocab, embedding_dimension):
    embedding_index = {}
    with zipfile.ZipFile(f) as z:
        with z.open("glove.6B.100d.txt") as f:
            n=0
            for line in f:
                if n:
                    values = line.split()
                    word = values[0]
                    if word in vocab: #only store words in current vocabulary 
                        coefs = np.asarray(values[1:], dtype='float32')
                        embedding_index[word] = coefs
                n+=1
    z.close()
    embedding_matrix = np.zeros((len(vocab) + 1, embedding_dimension))
    for word, i in vocab.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def createVocabCharNgrams(path, ngram_size):
    Vocab={}
    files = [filename for filename in listdir(path) if isfile(join(path, filename))]
    for filename in files:
        with open(filename, "r") as f:
            for line in f:
                for char_ngram in [''.join(ngram) for ngram in ngrams(list(line),ngram_size)]:
                    Vocab[char_nragm]=1
        f.close()
    return vocab
    


def segmentDocument(filename, nb_sentences_per_segment):
    words=[]
    lines=[]
    with open(filename, "r") as f:
         for line in f:
            words.extend(line.rstrip().split(" "))
            lines.append(line.rstrip())
    f.close()
    sentences=[]
    sentence=''
    docString=''
    for word in words:
        if re.match("^.*[\.\!\?]+$",word):
            sentence+=word+' '
            sentences.append(sentence)
            docString+=sentence
            sentence=''
        else:
            sentence+=word+' '
            
    segments=[sentences[i:i+nb_sentences_per_segment] for i in xrange(0,len(sentences),nb_sentences_per_segment)]
    
    return segments, docString


def segmentDocumentWords(filename, nb_words_per_segment):
    words=[]
    with open(filename, "r") as f:
         for line in f:
            tokens=line.rstrip().split(" ")
            for token in tokens:
                if token!='':
                    words.append(token)
    f.close()
    docString=' '.join(words)
    segments=[words[i:i+nb_words_per_segment] for i in xrange(0,len(words),nb_words_per_segment)]
    return segments, docString



def getLines(path):
    return open(path,"r").readlines()

    
def createTokenizer(pathTraining, pathTest):
    filesTraining = [join(pathTraining,filename) for filename in listdir(pathTraining) if isfile(join(pathTraining, filename))]
    filesTest = [join(pathTest,filename) for filename in listdir(pathTest) if isfile(join(pathTest, filename))]
    files=filesTraining+filesTest
    
    docs=[]
    segments=[]
    labelHash={}

    for file in files:
        match=re.match("^.+\/12([A-Z]+).+",file)
        if match:
            label=ord(match.group(1))-65
        else:
            print('Skipping filename:%s'%(file))
            continue
        lines=getLines(file)
        for line in lines:
            docs.append(line)
        if label not in labelHash:
            labelHash[label]=len(labelHash)

    tokenizer = Tokenizer() # <4>
    tokenizer.fit_on_texts(docs) # <5>

    nb_classes=len(labelHash)

    print nb_classes
    return (tokenizer,labelHash)    


from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot

def vectorizeDocumentsBOW(path, tokenizer, labelHash, nb_words_per_segment):
    files = [filename for filename in listdir(path) if isfile(join(path, filename))]
    docs=''
    segments=[]
    labels=[]

    for file in files:
        match=re.match("^12([A-Z]+).+",file)
        if match:
            label=ord(match.group(1))-65
        else:
            print('Skipping filename:%s'%(file))
            continue
        (segmented_document,docString)=segmentDocumentWords(join(path,file),nb_words_per_segment) # chunks of 100 words
        segments.extend(segmented_document)
        for segment in segmented_document:
            labels.append(label)

    vocab_len=len(set(docString.split()))
    
    labels=[labelHash[x] for x in labels]
    nb_classes=len(labelHash)
    
    X=[]
    y=[]
    i=0
    for segment in segments:
        segment=' '.join(segment)


#        X.append(pad_sequences([one_hot(segment,vocab_len+100)],nb_words_per_segment)[0])
        X.append(pad_sequences([hashing_trick(segment, round(vocab_len*1.3))], nb_words_per_segment)[0])
        i+=1
        
    y=np_utils.to_categorical(labels, nb_classes)

    return np.array(X),y



global LexH
LexH={}

def my_one_hot(d,n):
    global Lex
    words=d.split(" ")
    resA=[]
    for w in words:
        if w not in LexH:
            LexH[w]=len(LexH.keys())
        resA.append(LexH[w])
    return resA



    
def vectorizeDocumentsNgrams(path, ngram_size, tokenizer, labelHash):
    files = [filename for filename in listdir(path) if isfile(join(path, filename))]
    docs=[]
    segments=[]
    labels=[]

    for file in files:
        match=re.match("^12([A-Z]+).+",file)
        if match:
            label=ord(match.group(1))-65
        else:
            exit('Skipping filename:%s'%(file))
        (segmented_document,lines)=segmentDocument(join(path,file),10)
        for line in lines:
            docs.append(' '.join([''.join(ngram) for ngram in ngrams(line,ngram_size)]))
        segments.append(segmented_document)
        labels.append(label)

 #   tokenizer = Tokenizer() # <4>
 #   tokenizer.fit_on_texts(docs) # <5>

    labels=[labelHash[x] for x in labels]
    nb_classes=len(labelHash)
    X=[]
    for segment in segments:
        segment_text=[]
        for sentence in segment:
            sentence=' '.join([''.join(ngram) for ngram in ngrams(sentence,ngram_size)])
            segment_text.extend(sentence)
        X.append(tokenizer.texts_to_matrix(segment_text, mode='binary'))
    y=np_utils.to_categorical(labels, nb_classes)

    return X,y


def vectorizeDocumentsCharNgrams(path, ngram_size, tokenizer, labelHash):
    files = [filename for filename in listdir(path) if isfile(join(path, filename))]
    docs=[]
    segments=[]
    labels=[]

    for file in files:
        match=re.match("^12([A-Z]+).+",file)
        if match:
            label=ord(match.group(1))-65
        else:
            exit('Skipping filename:%s'%(file))
        (segmented_document,lines)=segmentDocument(join(path,file),10)
        for line in lines:
            docs.append(' '.join([''.join(ngram) for ngram in ngrams(list(line),ngram_size)]))
        segments.append(segmented_document)
        labels.append(label)

#    tokenizer = Tokenizer() # <4>
#    tokenizer.fit_on_texts(docs) # <5>

    labels=[labelHash[x] for x in labels]
    nb_classes=len(labelHash)
    X=[]
    for segment in segments:
        segment_text=[]
        for sentence in segment:
            sentence=' '.join([''.join(ngram) for ngram in ngrams(list(sentence),ngram_size)])
            segment_text.extend(sentence)
        X.append(tokenizer.texts_to_matrix(segment_text, mode='binary'))
    y=np_utils.to_categorical(labels, nb_classes)

    return X,y


def vectorizeDocumentsEmbedding_Words(path, embedding,embedding_size, tokenizer, labelHash):
    files = [filename for filename in listdir(path) if isfile(join(path, filename))]
    docs=[]
    segments=[]
    labels=[]

    for file in files:
        match=re.match("^12([A-Z]+).+",file)
        if match:
            label=ord(match.group(1))-65
        else:
            exit('Skipping filename:%s'%(file))
        (segmented_document,lines)=segmentDocument(join(path,file),10)
        for line in lines:
            docs.append(line)
        segments.append(segmented_document)
        labels.append(label)

#    tokenizer = Tokenizer() # <4>
#    tokenizer.fit_on_texts(docs) # <5>

    labels=[labelHash[x] for x in labels]
    nb_classes=len(labelHash)
    X=[]
    for segment in segments:
        segment_text=[]
        for sentence in segment:
            segment_text.extend(sentence)
        vect=np.zeros(embedding_size)
        n=0
        for word in segment_text.split(" "):
            if word in embedding.wv:
                vect=np.add(vect, embedding.wv[word])
                n+=1
        vect=np.divide(vect,n)
        X.append(vect)
    y=np_utils.to_categorical(labels, nb_classes)

    return X,y



def vectorizeDocumentsEmbedding_WordNgrams(path, embedding, embedding_size, ngram_size, max_vector_len, tokenizer, labelHash):
    files = [filename for filename in listdir(path) if isfile(join(path, filename))]
    docs=[]
    segments=[]
    labels=[]

    for file in files:
        match=re.match("^12([A-Z]+).+",file)
        if match:
            label=ord(match.group(1))-65
        else:
            exit('Skipping filename:%s'%(file))
        (segmented_document,lines)=segmentDocument(join(path,file),10)
        segments.append(segmented_document)
        doc_ngram_len=len(segmented_document)-ngram_size+1
        labels.append(label)

#    tokenizer = Tokenizer() # <4>
#    tokenizer.fit_on_texts(docs) # <5>

    labels=[labelHash[x] for x in labels]
    nb_classes=len(labelHash)
    X=[]
    for segment in segments:
        segment_text=[]
        for sentence in segment:
            segment_text.extend(' '.join(['+'.join(ngram) for ngram in ngrams(sentence,ngram_size)]))
        n=0
        vectA=[]
        for ngram in segment_text.split(" "):
            vect=np.zeros(embedding_size)
            for word in ngram.split("+"):
                if word in embedding.wv:
                    vect=np.add(vect, embedding.wv[word])
                    n+=1
            vect=np.divide(vect,n)
            vectA.append(vect)
        vectA=pad_sequences(vectA,maxlen=max_vector_len, padding='post')    
        X.append(vectA)
    y=np_utils.to_categorical(labels, nb_classes)

    return X,y
    


def vectorizeDocumentsEmbedding_CharNgrams(path, embedding, embedding_size, ngram_size, max_vector_len, tokenizer, labelHash):
    files = [filename for filename in listdir(path) if isfile(join(path, filename))]
    docs=[]
    segments=[]
    labels=[]

    for file in files:
        match=re.match("^12([A-Z]+).+",file)
        if match:
            label=ord(match.group(1))-65
        else:
            exit('Skipping filename:%s'%(file))
        (segmented_document,lines)=segmentDocument(join(path,file),10)
        segments.append(segmented_document)
        doc_ngram_len=len(segmented_document)-ngram_size+1
        labels.append(label)

#    tokenizer = Tokenizer() # <4>
#    tokenizer.fit_on_texts(docs) # <5>

    labels=[labelHash[x] for x in labels]
    nb_classes=len(labelHash)
    X=[]
    for segment in segments:
        segment_text=[]
        for sentence in segment:
            segment_text.extend(' '.join([''.join(ngram) for ngram in ngrams(list(sentence),ngram_size)]))
        n=0
        vectA=[]
        vect=np.zeros(embedding_size)
        for ngram in segment_text.split(" "):
            if ngram in embedding.wv:
                vect=np.add(vect, embedding.wv[word])
                n+=1
        vect=np.divide(vect,n)
        vectA.append(vect)
        vectA=pad_sequences(vectA,maxlen=max_vector_len, padding='post')    
        X.append(vectA)
    y=np_utils.to_categorical(labels, nb_classes)

    return X,y
    



