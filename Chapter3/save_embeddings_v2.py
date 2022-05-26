import codecs
import global_settings as G

def write_array_to_file(wf, array):
	for i in xrange(len(array)):
		wf.write(str(array.item(i)) + " ")
	wf.write("\n")

def save_embeddings(save_filepath, weights, vocabulary):
        rev = {v:k for k, v in vocabulary.iteritems()}
	with codecs.open(save_filepath, "w", "utf-8") as wf:
		wf.write(str(len(vocabulary)) + " " + str(weights.shape[1]) + "\n")
                for index in sorted(rev.iterkeys()):
                        word=rev[index]
			wf.write(word + " ")
			write_array_to_file(wf, weights[index])
