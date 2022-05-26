import sys
import re

def babify_gap(fnname):
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
                if sentence=="":
                    continue
                if first_name_offset>=offsetsH[i][0] and first_name_offset<=offsetsH[i][1]:
                    name_1=i
                if second_name_offset>=offsetsH[i][0] and second_name_offset<=offsetsH[i][1]:
                    name_2=i
                if pronoun_offset>=offsetsH[i][0] and pronoun_offset<=offsetsH[i][1]:
                    pron=i
                print "%d\t%s."%(i,sentence)
                i+=1

            
            if coref_first_name_pronoun=="TRUE":
                print "%d\tcorefers %s %s? \tyes\t%d %d"%(i,pronoun, first_name,name_1,pron)
            else:
                print "%d\tcorefers %s %s? \tno\t%d %d"%(i,pronoun, first_name,name_1,pron)
            i+=1
            if coref_second_name_pronoun=="TRUE":
                print "%d\tcorefers %s %s? \tyes\t%d %d"%(i,pronoun, second_name,name_2,pron)
            else:
                print "%d\tcorefers %s %s? \tno\t%d %d"%(i,pronoun, second_name,name_2,pron)

        n+=1
    f.close()

