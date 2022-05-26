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


def mergeDictionaries(d1, d2):
    d = d1.copy()
    d.update(d2)
    return d

def segmentDocumentWords(filename, nb_words_per_segment):
    wordsDict={}
    words=[]
    with open(filename, "r") as f:
         for line in f:
            tokens=line.rstrip().split(" ")
            for token in tokens:
                if token!='':
                    words.append(token)
                    wordsDict[token]=1
                    
    f.close()
    segments=[words[i:i+nb_words_per_segment] for i in xrange(0,len(words),nb_words_per_segment)]
    return segments, wordsDict


def segmentDocumentNgrams(filename, nb_words_per_segment, ngram_size):
    wordsDict={}
    words=[]
    with open(filename, "r") as f:
         for line in f:
            ngrams_list=ngrams(line.rstrip(),ngram_size)
            for ngram in ngrams_list:
                joined='_'.join(ngram)
                words.append(joined)
                wordsDict[joined]=1
    f.close()
    segments=[words[i:i+nb_words_per_segment] for i in xrange(0,len(words),nb_words_per_segment)]
    return segments, wordsDict


def segmentDocumentCharNgrams(filename, nb_words_per_segment, ngram_size):
    wordsDict={}
    words=[]
    with open(filename, "r") as f:
         for line in f:
            line=line.rstrip().replace(' ','#')
            char_ngrams_list=ngrams(list(line),ngram_size)
            for char_ngram in char_ngrams_list:
                joined=''.join(char_ngram)
                words.append(joined)
                wordsDict[joined]=1
    f.close()
    segments=[words[i:i+nb_words_per_segment] for i in xrange(0,len(words),nb_words_per_segment)]
    return segments, wordsDict




def getLines(path):
    return open(path,"r").readlines()

    
def createTokenizer(pathTraining, pathTest):
    filesTraining = [join(pathTraining,filename) for filename in listdir(pathTraining) if isfile(join(pathTraining, filename))]
    filesTest = [join(pathTest,filename) for filename in listdir(pathTest) if isfile(join(pathTest, filename))]
    files=filesTraining+filesTest
    
    docs=[]
    segments=[]
    labelDict={}

    for file in files:
        match=re.match("^.*\/?12[A-Z][a-z]+([A-Z]+).+",file)
        if match:
            label=ord(match.group(1))-65
        else:
            print('Skipping filename:%s'%(file))
            continue
        lines=getLines(file)
        for line in lines:
            docs.append(line)
        if label not in labelDict:
            labelDict[label]=len(labelDict)

    tokenizer = Tokenizer() # <4>
    tokenizer.fit_on_texts(docs) # <5>

    nb_classes=len(labelDict)

    print nb_classes
    return (tokenizer,labelDict)    

def createLabelDict(pathTraining, pathTest):
    filesTraining = [join(pathTraining,filename) for filename in listdir(pathTraining) if isfile(join(pathTraining, filename))]
    filesTest = [join(pathTest,filename) for filename in listdir(pathTest) if isfile(join(pathTest, filename))]
    files=filesTraining+filesTest

    labelDict={}

    for file in files:
        match=re.match("^.*\/?12[A-Z][a-z]+([A-Z]+).+",file)
        if match:
            label=ord(match.group(1))-65
        else:
            print('Skipping filename:%s'%(file))
            continue
        if label not in labelDict:
            labelDict[label]=len(labelDict)

    return labelDict



def createLabelDictOneFile(path):
    files = [join(path,filename) for filename in listdir(path) if isfile(join(path, filename))]
    labelDict={}

    for file in files:
        match=re.match("^.*\/?12[A-Z][a-z]+([A-Z]+).+",file)
        if match:
            label=ord(match.group(1))-65
        else:
            print('Skipping filename:%s'%(file))
            continue
        if label not in labelDict:
            labelDict[label]=len(labelDict)

    return labelDict   



def createTokenizerOneFile(path):
    files = [join(path,filename) for filename in listdir(path) if isfile(join(path, filename))]
    
    docs=[]
    segments=[]
    labelDict={}

    for file in files:
        match=re.match("^.*\/?12[A-Z][a-z]+([A-Z]+).+",file)
        if match:
            label=ord(match.group(1))-65
        else:
            print('Skipping filename:%s'%(file))
            continue
        lines=getLines(file)
        for line in lines:
            docs.append(line)
        if label not in labelDict:
            labelDict[label]=len(labelDict)

    tokenizer = Tokenizer() # <4>
    tokenizer.fit_on_texts(docs) # <5>

    nb_classes=len(labelDict)

    print nb_classes
    return (tokenizer,labelDict)    


from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot

#def vectorizeDocumentsBOW(path, tokenizer, labelHash, nb_words_per_segment):
def vectorizeDocumentsBOW(path, labelDict, nb_words_per_segment):
    files = [filename for filename in listdir(path) if isfile(join(path, filename))]
    segments=[]
    labels=[]
    globalDict={}
    
    for file in files:
        match=re.match("^.*12[A-Z][a-z]+([A-Z]+).+",file)
        if match:
            label=ord(match.group(1))-65
        else:
            print('Skipping filename:%s'%(file))
            continue

        (segmented_document,wordDict)=segmentDocumentWords(join(path,file),nb_words_per_segment) # chunks of 100 words

        globalDict=mergeDictionaries(globalDict,wordDict)

        segments.extend(segmented_document)

        for segment in segmented_document:
            labels.append(label)

    vocab_len=len(globalDict)
    
    labels=[labelDict[x] for x in labels]
    nb_classes=len(labelDict)
    
    X=[]
    y=[]

    for segment in segments:
        segment=' '.join(segment)
        X.append(pad_sequences([hashing_trick(segment, round(vocab_len*1.5))], nb_words_per_segment)[0])
        
    y=np_utils.to_categorical(labels, nb_classes)

    return np.array(X), y, int(vocab_len*1.5)+1

    
#def vectorizeDocumentsNgrams(path, ngram_size, tokenizer, labelHash, nb_words_per_segment):
def vectorizeDocumentsNgrams(path, ngram_size, labelDict, nb_words_per_segment):
    files = [filename for filename in listdir(path) if isfile(join(path, filename))]
    segments=[]
    labels=[]
    globalDict={}
    
    for file in files:
        match=re.match("^.*12[A-Z][a-z]+([A-Z]+).+",file)
        if match:
            label=ord(match.group(1))-65
        else:
            print('Skipping filename:%s'%(file))
            continue
        (segmented_document,wordDict)=segmentDocumentNgrams(join(path,file),nb_words_per_segment, ngram_size)

        globalDict=mergeDictionaries(globalDict,wordDict)
        
        segments.extend(segmented_document)
        for segment in segmented_document:
            labels.append(label)

    vocab_len=len(globalDict)
    
    labels=[labelDict[x] for x in labels]
    nb_classes=len(labelDict)
    
    X=[]
    y=[]

    for segment in segments:
        segment=' '.join(segment)
        X.append(pad_sequences([hashing_trick(segment, round(vocab_len*1.5))], nb_words_per_segment)[0])
    
        
    y=np_utils.to_categorical(labels, nb_classes)

    return np.array(X),y, int(vocab_len*1.5)+1


#def vectorizeDocumentsCharNgrams(path, ngram_size, tokenizer, labelHash, nb_words_per_segment):
def vectorizeDocumentsCharNgrams(path, ngram_size, labelDict, nb_words_per_segment):
    files = [filename for filename in listdir(path) if isfile(join(path, filename))]
    segments=[]
    labels=[]
    globalDict={}

    for file in files:
        match=re.match("^.*12[A-Z][a-z]+([A-Z]+).+",file)
        if match:
            label=ord(match.group(1))-65
        else:
            print('Skipping filename:%s'%(file))
            continue
        
        (segmented_document,wordDict)=segmentDocumentCharNgrams(join(path,file),nb_words_per_segment, ngram_size)

        globalDict=mergeDictionaries(globalDict,wordDict)

        segments.extend(segmented_document)
        for segment in segmented_document:
            labels.append(label)

    vocab_len=len(globalDict)
    
    labels=[labelDict[x] for x in labels]
    nb_classes=len(labelDict)
    
    X=[]
    y=[]

    for segment in segments:
        segment=' '.join(segment)
        X.append(pad_sequences([hashing_trick(segment, round(vocab_len*1.5))], nb_words_per_segment)[0])
        
    y=np_utils.to_categorical(labels, nb_classes)

    return np.array(X),y, (vocab_len*1.5)+1



#def vectorizeDocumentsEmbedding_Words(path, nb_words_per_segment, embedding, embedding_size, tokenizer, labelHash):
def vectorizeDocumentsEmbedding_Words(path, nb_words_per_segment, embedding, embedding_size, labelDict):
    files = [filename for filename in listdir(path) if isfile(join(path, filename))]
    segments=[]
    labels=[]

    for file in files:
        match=re.match("^.*12[A-Z][a-z]+([A-Z]+).+",file)
        if match:
            label=ord(match.group(1))-65
        else:
            print('Skipping filename:%s'%(file))
            continue
        (segmented_document,docString)=segmentDocumentWords(join(path,file),nb_words_per_segment)
        segments.extend(segmented_document)
        for segment in segmented_document:
            labels.append(label)

    vocab_len=len(set(docString.split()))    
    labels=[labelDict[x] for x in labels]
    nb_classes=len(labelDict)
    
    X=[]
    y=[]

    for segment in segments:
        vect=np.zeros(embedding_size)
        n=0
        for word in segment:
            word=word.lower()
            if word in embedding: #.wv:
                vect=np.add(vect, embedding[word])                
                n+=1
        if n>0:
            vect=np.divide(vect,float(n))
        else:
            print "No match for",segment
        X.append(vect)
    y=np_utils.to_categorical(labels, nb_classes)

    return X,y



#def vectorizeDocumentsEmbedding_WordNgrams(path, embedding, embedding_size, ngram_size, max_vector_len, tokenizer, labelHash):
def vectorizeDocumentsEmbedding_WordNgrams(path, embedding, embedding_size, ngram_size, max_vector_len, labelDict):
    files = [filename for filename in listdir(path) if isfile(join(path, filename))]
    docs=[]
    segments=[]
    labels=[]

    for file in files:
        match=re.match("^.*12[A-Z][a-z]+([A-Z]+).+",file)
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

    labels=[labelDict[x] for x in labels]
    nb_classes=len(labelDict)
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
    


def vectorizeDocumentsEmbedding_CharNgrams(path, embedding, embedding_size, ngram_size, max_vector_len, labelDict):
    files = [filename for filename in listdir(path) if isfile(join(path, filename))]
    docs=[]
    segments=[]
    labels=[]

    for file in files:
        match=re.match("^.*12[A-Z][a-z]+([A-Z]+).+",file)
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

    labels=[labelDict[x] for x in labels]
    nb_classes=len(labelDict)
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
    



