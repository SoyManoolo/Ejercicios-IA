import idx2numpy
import numpy as np
from model import model

class data:

    # Estos atributos debo inicializarlos en el main pasandoles este mismo valor (debo crear un constructor)
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = np.eye(10)[y_train]
        self.y_test = np.eye(10)[y_test]
        pass

    # En esta funcion debo dividir con reshape las x entrenadas en n cantidad para no entrenar todas directamentes (ineficaz)
    # Por lo tanto tendré diferentes etapas y cada etapa será un entrenamiento para el modelo, haciendo que en cada etapa actualice las w y B de cada capa
    # Entonces aqui creo un modelo y llamo a la funcion forward del modelo por cada etapa hecha, calculo el coste y luego llamo al backward
    def trainloop(self, m: model):
        # Hago un reshape de los datos de entreno para tener los datos como un solo vector, haciendo que sean indexables (estos datos se siguen tratando de la misma manera)
        x_train_batched = self.x_train.reshape(60000, 784)
        # Hago un loop en el que en cada iteracion se hace un entreno con 500
        for i in range(0, len(x_train_batched), 500):
            x_batch = x_train_batched[i:i+500]
            y_batch = self.y_train[i:i+500]

            y = m.forward(x_batch)
            loss = m.loss(y, y_batch)
            m.backward(loss)
        return loss