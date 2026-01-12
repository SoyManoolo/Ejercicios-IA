from hiddenlayer import hiddenlayer
from act import act
import numpy as np

class model:
    w_shape: list
    b_shape: list
    list_hidden_layers: list[hiddenlayer]

    # Le paso una lista de tuplas para inicializar w y una lista de ints para b en cada capa oculta que se vaya a crear
    def __init__(self, w_shape: list, b_shape: list):
        self.w_shape = w_shape
        self.b_shape = b_shape
        self.list_hidden_layers = []

        # Inicializo las capas ocultas a partir de un bucle y las guardo en la lista
        for i in range(len(w_shape)):
            self.list_hidden_layers.append(hiddenlayer(w_shape[i], b_shape[i]))
        pass

    # En esta función obtengo la y predicha haciendo un forward por cada layer que tengo
    # Paso por la hidden_layer (esto ya es una y, pero no la final o predicha que busco)
    # Necesito pasar la cantidad de capas para que se repita, tambien los datos de entreno
    # Luego aplico la funcion de activacion para obtener la siguiente N en caso de que hayan mas capas o la y predicha si ya no hay mas capas
    def forward(self, x: np.ndarray):
        # Recorro la lista de capas ocultas para hacer el calculo en cada capa oculta
        for hidden_layer in self.list_hidden_layers:
            N = hidden_layer.forward(x)
            x = N
        return N

    #En esta funcion obtengo los dw, dB y dN para hacer update de w y b a las hidden_layer 
    def backward(self, loss_gradient):
        # Recorro la lista de capas ocultas desde el final
        for hidden_layer in reversed(self.list_hidden_layers):
            loss_gradient = hidden_layer.backward(loss_gradient)
        pass

    # En esta funcion obtengo el coste o loss del modelo, el cual debo propagar a las capas anteriores para ir calculando las dw, dB y dN / y (esta y se sigue propagando atrás)
    def loss(self, y: np.ndarray, y_train: np.ndarray):
        return (y - y_train)
