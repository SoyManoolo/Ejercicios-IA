import hiddenlayer, act
import numpy as np

class layer:
    label: hiddenlayer
    act: act

    def __init__(self, input, output):
        w = np.random.rand()
        b = np.random.rand()
        pass