import numpy as np

class act:

    @staticmethod
    def funcio_act(y):
        return np.maximum(0, y)

    @staticmethod
    def dFuncio_act(x):
        x[x<=0] = 0
        x[x>0] = 1
        return x