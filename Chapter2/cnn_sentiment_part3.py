model.add(Convolution1D(64, 3, padding="same"))
model.add(Convolution1D(32, 3, padding="same"))
model.add(Convolution1D(16, 3, padding="same"))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(2,activation='sigmoid'))

