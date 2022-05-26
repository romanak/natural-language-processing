import sys
import re


def splitDocument(filename, nb_sentences_per_segment):
    words=[]
    with open(filename, "r") as f:
        for line in f:
            words.extend(line.split())
    sentences=[]
    sentence=[]
    for word in words:
        if re.match("^.*[\.\!\?]+$",word):
            sentence.append(word)
            sentences.append(sentence)
            sentence=[]
        else:
            sentence.append(word)
    return zip(*[sentences[i::nb_sentences_per_segment] for i in range(nb_sentences_per_segment)]) 


if __name__=="__main__":
    s=splitDocument(sys.argv[1],5)
    print s[0]

    
