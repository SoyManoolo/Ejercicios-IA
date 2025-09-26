from typing import List, Dict
import math

bs_data = [
        {"x": 1, "y": 1},
        {"x": "some", "y": 12},
        {"x": 3, "y": 9},
        {"x": 9, "y": 4},
        {"x": 1, "y": 1},
        {"x": 1, "y": 5},
        {"x": 5, "y": 2},
        {"x": 4, "y": 10},
        {"x": 8, "y": 8},
        {"x": -3, "y": 2.3}
        ]

bs_data_clean = []

distance = 0

best = float('inf')

# Acepte una lista de diccionarios con una serie de coordenadas ("x" e "y") con números enteros
# Verifique que todo son números enteros (en caso de no ser así que elimine las coordenadas que no lo cumplan)
# Calcule el orden que optimice la distancia mínima a recorrer si tenemos que pasar por todos los puntos.
# Muestre por pantalla la distancia mínima obtenida y devuelva un diccionario con el orden correcto de los puntos.

# Funcion para limpiar la lista y que solo queden coordenadas con numeros enteros - LISTO
def clean_list(lista:List):
    for point in lista: # Recorro la lista
        if type(point["x"]) == int and type(point["y"]) == int: # Compruebo de si los valores son enteros
            bs_data_clean.append(point) # Si es entero lo introduzco en la nueva lista
    bus_stops(bs_data_clean)

# Funcion que calcula la distancia entre dos puntos - Revisar la extraccion de coordenadas
def calc_distance(point1: Dict, point2: Dict):
    x1 = point1['x']
    y1 = point1['y']
    x2 = point2['x']
    y2 = point2['y']
    print(x1,y1,x2,y2)
    return math.sqrt(math.pow(x2-x1,2)+math.pow(y2-y1,2)) # Calculo la distancia entre estos dos puntos y lo devuelvo


def bus_stops(lista: List):
    queda: List = bs_data_clean.copy() # Copio la lista para no modificar la original
    for actual_point in lista:
        print(queda)
        queda.remove(actual_point)
        best_way(actual_point, queda)
    #print(lista)

# Funcion que mira cual es la mejor opcion entre diferentes puntos
def best_way(actual_point: Dict, queda: List):
    print(queda)
    for point in queda: # Recorro todas las coordenadas restantes y calculo las distancias con el punto actual
        queda.remove(point)
        print(calc_distance(actual_point, point))
        best_way(point, queda)
    return

clean_list(bs_data)