from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.layers.core import Dense, Activation, Dropout
# ... process data 
model = Sequential()
model.add(Dense(128, input_dim=input_dim))
model.add(Activation('relu'))

