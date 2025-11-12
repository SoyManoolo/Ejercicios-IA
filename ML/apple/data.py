import idx2numpy
import numpy as np

x_train = idx2numpy.convert_from_file('./train-images.idx3-ubyte')
y_train = idx2numpy.convert_from_file('./train-labels.idx1-ubyte')
x_test = idx2numpy.convert_from_file('./t10k.images.idx3-ubyte')
y_test = idx2numpy.convert_from_file('./t10k-labels.idx1-ubyte')

print(np.reshape(x_train))