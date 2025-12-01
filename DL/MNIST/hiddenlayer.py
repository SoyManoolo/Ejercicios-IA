import numpy as np
from act import act

class hiddenlayer:
    x: np.ndarray
    w: np.ndarray
    b: np.ndarray
    dw: np.ndarray
    db: np.ndarray
    z: np.ndarray
    N: np.ndarray
    alpha: float

    # Pido la tupla de w para crear una matriz de pesos iniciales de la medida que me diga la tupla
    def __init__(self, w_tuple: list, b: int):
        self.w = np.random.rand(*w_tuple)
        self.b = np.random.rand(b)
        self.alpha = 0.01

    # Esta funcion sirve para calcular z / y a partir de xw + B o Nw + B donde x y N son un vector, w es una matriz y B es un vector
    def forward(self, x: np.ndarray):
        self.x = x
        self.z = x@self.w + self.b
        self.N = act.funcio_act(self.z)
        return self.N
    
    # dy es el coste que se quiere propagar a esta capa y devolvemos
    def backward(self, err):
        err = err*act.dFuncio_act(self.z)
        self.dw = self.x.T@err
        self.db = np.sum(err, axis=0)
        self.update()
        return err @ self.w.T

    # Esta funcion sirve para actualizar las w o B de la capa (layer) e ir reduciendo el coste o loss del modelo
    def update(self):
        self.w = self.w - self.alpha*self.dw
        self.b = self.b - self.alpha*self.db
        return