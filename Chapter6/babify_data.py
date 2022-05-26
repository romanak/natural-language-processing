import sys
import re

def babify_pp(fname):
    inp=open(fname,"r")
    for line in inp:
        m=re.match("^(.+),([^,]+)$",line.rstrip())
        if m:
            features=m.group(1).split(",")
            label=m.group(2)
            n=1
            print "1 %s V." %(features[0])
            print "2 %s N." %(features[1])
            pp_str=features[2] +' ' + features[3]
            print "%d attach %s? \t%s\t%s" % (n,pp_str,label, ' '.join([str(x) for x in range(1,3)]))
    inp.close()



def babify_pp2(fname):
    inp=open(fname,"r")
    for line in inp:
        m=re.match("^(.+),([^,]+)$",line.rstrip())
        if m:
            features=m.group(1).split(",")
            label=m.group(2)
            n=1
            print "1 %s." %(' '.join(features))
            pp_str=features[2] +' ' + features[3]
            print "2 attach %s? \t%s\t1" % (pp_str,label)
    inp.close()


    
def babify_gap_naive(fname):
    f=open(fname,"r")
    n=0
    for line in f:
        if n>0:
            feat=line.split("\t")
            text=feat[1]
            pronoun=feat[2]
            pronoun_offset=int(feat[3])
            first_name=feat[4]
            first_name_offset=int(feat[5])
            coref_first_name_pronoun=feat[6]
            second_name=feat[7]
            second_name_offset=int(feat[8])
            coref_second_name_pronoun=feat[9]
            
            sentences=text.split(".")

            offsetsH={}
            start=0
            i=1
            for sentence in sentences:
                end=start+len(sentence)
                offsetsH[i]=(start,end)
                start=end+1
                i+=1

            i=1
            for sentence in sentences:
                sentence=re.sub("^\s+","",sentence)
                if sentence=="":
                    next
                if first_name_offset>=offsetsH[i][0] and first_name_offset<=offsetsH[i][1]:
                    name_1=i
                if second_name_offset>=offsetsH[i][0] and second_name_offset<=offsetsH[i][1]:
                    name_2=i
                if pronoun_offset>=offsetsH[i][0] and pronoun_offset<=offsetsH[i][1]:
                    pron=i
                print "%d %s."%(i,sentence)
                i+=1

            
            if coref_first_name_pronoun=="TRUE":
                print "%d corefers %s %s? \tyes\t%d %d"%(i,pronoun, first_name,name_1,pron)
            else:
                print "%d corefers %s %s? \tno\t%d %d"%(i,pronoun, first_name,name_1,pron)
            i+=1
            if coref_second_name_pronoun=="TRUE":
                print "%d corefers %s %s? \tyes\t%d %d"%(i,pronoun, second_name,name_2,pron)
            else:
                print "%d corefers %s %s? \tno\t%d %d"%(i,pronoun, second_name,name_2,pron)

        n+=1
    f.close()



import string
def strip_punc(s):
    s=s.translate(None, string.punctuation)    
    s=re.sub(" +"," ",s)
    s=re.sub("^\s+","",s)
    s=re.sub("\s+$","",s)
    return s

def trim_spaces(arr):
    result=[]
    for s in arr:
        if s=='':
            continue
        s=re.sub("^\s+","",s)
        s=re.sub("^\s+$","",s)
        result.append(s)
    return result
    

def babify_gap_gender(fname):
    from gender_detector import GenderDetector # sudo pip install gender_detector
    detector = GenderDetector('us')
    import nltk
    # detector.guess('Shelley')

    f=open(fname,"r")
    n=0
    for line in f:
        if n>0:
            feat=line.rstrip().split("\t")
            text=feat[1]
            pronoun=feat[2]
            pronoun_offset=int(feat[3])
            first_name=feat[4]
            first_name_offset=int(feat[5])
            coref_first_name_pronoun=feat[6]
            second_name=feat[7]
            second_name_offset=int(feat[8])
            coref_second_name_pronoun=feat[9]

            sentences=text.rstrip().split(".")
            sentences=trim_spaces(sentences)
            
            offsetsH={}
            start=0
            i=1
            
            for sentence in sentences:
                end=start+len(sentence)+1 # one for period
                offsetsH[i]=(start,end)
                start=end+1
                i+=1
            i=1
            for sentence in sentences:
                stripped_sentence=strip_punc(sentence)
                if re.match(" *$",stripped_sentence):
                    continue
                sentence_tagged=nltk.pos_tag(stripped_sentence.split(" "))
                sentence=""
                for (word, pos) in sentence_tagged:
                    if pos=='NNP':
                        try:
                            gender=detector.guess(word)
                        except:
                            gender='unknown'
                        sentence+=" %s %s"%(word, gender)
                    elif re.match("^PRP.*",pos):
                        sentence+=" %s"%(word)
                    elif re.match("^VB.+",pos):
                        sentence+=" %s"%(word)                
                if first_name_offset>=offsetsH[i][0] and first_name_offset<=offsetsH[i][1]:
                    name_1=i
                if second_name_offset>=offsetsH[i][0] and second_name_offset<=offsetsH[i][1]:
                    name_2=i
                if pronoun_offset>=offsetsH[i][0] and pronoun_offset<=offsetsH[i][1]:
                    pron=i
                if sentence=='':
                    print "%d %s."%(i,' '.join([w[0] for w in sentence_tagged]))
                else:
                    print "%d%s."%(i,sentence)
                i+=1


            if pron==i:
                print "ERROR:",sentences
                print pron,pronoun
                print pronoun_offset
                print offsetsH
                exit(0)
            if coref_first_name_pronoun=="TRUE":
                print "%d corefers %s %s? \tyes\t%d %d"%(i,pronoun, first_name,name_1,pron)
            else:
                print "%d corefers %s %s? \tno\t%d %d"%(i,pronoun, first_name,name_1,pron)
            i+=1
            if coref_second_name_pronoun=="TRUE":
                print "%d corefers %s %s? \tyes\t%d %d"%(i,pronoun, second_name,name_2,pron)
            else:
                print "%d corefers %s %s? \tno\t%d %d"%(i,pronoun, second_name,name_2,pron)
            if pron==i:
                print "ERROR:",sentences
                print pron,pronoun
                print pronoun_offset
                print offsetsH
                exit(0)

        n+=1
    f.close()



def babify_dimin(fname):
    f=open(fname,"r")
    for line in f:
        features=line.rstrip().split(",")
        label=features.pop()
        i=1
        for feature in features:
            if feature=="=":
                feature="is"
            if feature =="-":
                feature="dash"
            if feature=="+":
                feature="plus"
            if feature=="@":
                feature="schwa"
            print "%d %s."%(i,feature)
            i+=1
        print "%d suffix? \t%s\t%s"%(i,label,' '.join([str(x) for x in range(1,i)]))
    f.close()


def babify_dimin2(fname):
    f=open(fname,"r")
    for line in f:
        features=line.rstrip().split(",")
        label=features.pop()
        fA=[]
        for feature in features:
            if feature=="=":
                feature="eq"
            elif feature =="-":
                feature="dash"
            elif feature=="+":
                feature="plus"
            elif feature=="@":
                feature="schwa"
            elif feature=='{':
                feature="lbr"
            elif feature=='}':
                feature="rbr"
            fA.append(feature)    
#        print "1 %s."%(' '.join(fA[0:6]))
 #       print "2 %s."%(' '.join(fA[6:12]))
  #      print "3 suffix? \t%s\t%s"%(label,"1 2")

        print "1 %s."%(' '.join(fA))
        print "2 suffix %s? \t%s\t%s"%(fA[-1],label,"1")


  
    f.close()


def ngrams(s,n):
    s=s.split()
    ngrs=[s[i:i+n] for i in xrange(len(s)-n+1)]
    return ngrs



def babify_conll02(fname):
    f=open(fname,"r")
    Lex={}
    for line in f:
        if re.match("DOCSTART",line):
            continue
        m=re.match("^([^\s]+)\s+([^\s]+)\s+(.+)$",line.rstrip())
        if m:
            word=m.group(1)
            pos=m.group(2)
            if word in Lex:
                if pos not in Lex[word]:
                    Lex[word].append(pos)
            else:
                Lex[word]=[pos]
    f.seek(0)

    ngramsize=3
    focus=1
    story=""
    for line in f:
        if re.match(".+DOCSTART.+",line):
            continue
        if re.match("^\s*$",line.rstrip()):
            ngrs=ngrams(story,ngramsize)
            n=1
            ambig=False
            for ngr in ngrs:
                fact="%d"%(n)
                i=0
                for w in ngr:
                    word_plus_pos=w.split("#")
                    word=word_plus_pos[0]
                    pos=word_plus_pos[1]
                    lex_pos='_'.join(Lex[word])
                    if i==focus:
                        #fact+=" %s %s"%(word,lex_pos)
                        fact+=" %s"%(lex_pos)
                        if '_' in lex_pos:
                            ambig=True
                            unique_pos=pos
                            ambig_word=word
                            ambig_pos=lex_pos
                    elif i==ngramsize-1:
                        #fact+=" %s %s."%(word,lex_pos)
                        fact+=" %s."%(lex_pos)
                        print fact
                    else:
                        #fact+=" %s %s"%(word,lex_pos)
                        fact+=" %s"%(lex_pos)
                    i+=1
                if ambig:
                    n+=1
                    ambig=False
                    #print "%d pos %s? \t%s\t%d"%(n,ambig_word,unique_pos,n-1)
                    if n>2:
                        print "%d pos %s? \t%s\t%d %d"%(n,ambig_pos,unique_pos,n-2,n-1)
                    else:
                        print "%d pos %s? \t%s\t%d"%(n,ambig_pos,unique_pos,n-1)
                    n=0

                n+=1
            story=""
        else:
            m=re.match("^([^\s]+)\s+([^\s]+)\s+(.+)$",line.rstrip())
            if m:
                story+=m.group(1)+"#"+m.group(2)+" "
    f.close()
    exit(0)




                                     
                                                      
    
    
if __name__=="__main__":
    babify_pp2(sys.argv[1])
    #babify_gap_gender(sys.argv[1])
    #babify_dimin(sys.argv[1])
    #babify_conll02(sys.argv[1])
#    babify_dimin2(sys.argv[1])
    
            
