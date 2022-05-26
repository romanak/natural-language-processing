data = pd.read_csv(sys.argv[1],sep='\t') # tsv file
max_words = 1000
tokenizer = Tokenizer(num_words=max_words, split=' ')

tokenizer.fit_on_texts(data['text'].values)

X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)
Y = pd.get_dummies(data['label']).values

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 36)

