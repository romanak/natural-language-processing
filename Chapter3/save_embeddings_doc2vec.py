import codecs
import global_settings as G

def write_array_to_file(wf, array):
	for i in xrange(len(array)):
		wf.write(str(array.item(i)) + " ")
	wf.write("\n")

def save_embeddings(save_filepath, weights, nb_docs):
	with codecs.open(save_filepath, "w", "utf-8") as wf:
		wf.write(str(nb_docs) + " " + str(weights.shape[1]) + "\n")
                for index in range(nb_docs):
			wf.write("doc_"+str(index) + " ")
			write_array_to_file(wf, weights[index])
