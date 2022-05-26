model.add(Convolution1D(64, 3, padding="same"))
print model.output_shape
model.add(Convolution1D(32, 3, padding="same"))
print model.output_shape
model.add(Convolution1D(16, 3, padding="same"))
print model.output_shape

