embedding_vector_length = 100

model = Sequential()

model.add(Embedding(max_words, embedding_vector_length, input_length=X.shape[1]))

