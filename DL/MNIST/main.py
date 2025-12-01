import idx2numpy
import numpy as np
from model import model
from data import data

w_shape = [(784, 32), (32, 10)]
b_shape = [32, 10]

m = model(w_shape, b_shape)

x_train = idx2numpy.convert_from_file('./train-images.idx3-ubyte')
y_train = idx2numpy.convert_from_file('./train-labels.idx1-ubyte')
x_test = idx2numpy.convert_from_file('./t10k-images.idx3-ubyte')
y_test = idx2numpy.convert_from_file('./t10k-labels.idx1-ubyte')

d = data(x_train, x_test, y_train, y_test)

print(d.trainloop(m))