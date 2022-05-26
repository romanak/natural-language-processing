import argparse
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Lambda, LSTM, Dropout, BatchNormalization, Activation


def splitData(X,y):
    X,y=shuffle(X,y,random_state=42)
    AuthorsX={}

    for (x,y) in zip(X,y):
        if y in AuthorsX:
            AuthorsX[y].append(x)
        else:
            AuthorsX[y]=[x]

    max_samples_per_author=10

    X_left=[]
    X_right=[]
    y_lr=[]

    for author in AuthorsX:
        nb_texts=len(AuthorsX[author])
        nb_samples=min(nb_texts, max_samples_per_author)

        left=np.take(AuthorsX[author],random.sample(range(0, nb_samples), nb_samples))
        for other_author in AuthorsX:
            nb_samples_other=min(len(AuthorsX[other_author]), max_samples_per_author)
            right=np.take(AuthorsX[author],random.sample(range(0, nb_samples_other), nb_samples_other))
            for (l,r) in zip(left,right):
                X_left.append(l)
                X_right.append(r)            
                if author==other_author:
                    y_lr.append(1.0)
                else:
                    y_lr.append(0.0)
    return X_left,X_right,y_lr    


class CNNModel(object):
    def create(self, vocab_size=500, max_length=300):
        x = Input(shape=(max_length,))
        embedded = Embedding(vocab_size, 64, input_length=max_length)(x)
        self.cnn = Model(inputs=embedded, output=self._predict(embedded))
        self.cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
                
    def predict(self, encoded):
            h=Dense(300, activation='relu')(encoded)
            h=Convolution1D(32, 30, padding="same")(h)
            h=Flatten()(h)
            return Dense(1, activation='sigmoid', name='pred')(h)

    
def exponent_neg_manhattan_distance(left, right):
        return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


class SiameseModel(object):
        def create(self,X_left, X_right,y_lr):
                self.shared_model = CNNModel()
                self.shared_model.create(vocab_size=NUM_WORDS, max_length=MAX_LENGTH)

                left_input = Input(shape=(max_seq_length,), dtype='int32')
                right_input = Input(shape=(max_seq_length,), dtype='int32')

                leftPred=CNNModel(left_input)
                rightPred=CNNModel(right_input)
        
                distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([leftPred, rightPred])

                self.siameseModel = Model([leftPred, rightPred], [distance])
                self.siameseModel.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
                self.siameseModel.fit([X_left, X_right], y_lr, batch_size=batch_size, nb_epoch=n_epoch,
                            validation_split=0.3)

       def predict(self, X_left, X_right, y_lr):
               return self.siameseModel.predict(X_left, X_right, y_lr)
               


       
if __name__=="__main__":
        
    train="/data/pan12-authorship-attribution-training-corpus-2012-03-28/"
    test="/data/pan12-authorship-attribution-test-corpus-2012-05-24/GT/"
    
    nb_epochs=10

    (tokenizer, labelHash)=createTokenizer(train,test)

    input_dim = 500 # word chunks
    
    X, y=vectorizeDocumentsBOW(train,tokenizer,labelHash,input_dim)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    X_train_left, X_train_right, y_train_lr=splitData(X,y)
    X_test_left, X_test_right, y_test_lr=splitData(X,y)
    
    siamese=SiameseModel(X_train_left, X_train_right, y_train_lr)

    pred=siamese.predict(X_test_left, X_test_right, y_test_lr)
