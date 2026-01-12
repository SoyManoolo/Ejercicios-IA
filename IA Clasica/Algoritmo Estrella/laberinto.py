import pyamaze as maze
from queue import PriorityQueue
import time

# Parchear pyamaze para evitar el error 'zoomed' en Linux
import tkinter as tk
original_state = tk.Tk.state
def patched_state(self, newstate=None):
    if newstate == 'zoomed':
        return  # Ignorar 'zoomed' en Linux
    return original_state(self, newstate)
tk.Tk.state = patched_state

# Declaro la cantidad de filas y columnas que tendrá el laberinto
ROWS = 100
COLS = 100

# Creo el laberinto
m=maze.maze(ROWS,COLS)
m.CreateMaze(theme='light')

start = (ROWS, COLS)
goal = (1,1)

# Función donde se guardan las celdas a vecinas accesibles junto con el g_score (celdas recorridas), más la distáncia heurística y la celda padre en la lista celdas vecinas
def expandir_celda(nodo, m: maze.maze, celdas_vecinas: list, g_scores, padres):
    celda, g, h, padre = nodo
    x, y = celda
    
    # Si la celda vecina a donde me encuentro NO tiene una pared/muro (1) entonces la añado a la lista
    if m.maze_map[celda]['S'] == 1:
        vecino = (x+1, y)
        nuevo_g = g + 1
        # Compruebo si el nuevo camino es mejor que el que ya teniamos
        if vecino not in g_scores or nuevo_g < g_scores[vecino]:
            g_scores[vecino] = nuevo_g
            padres[vecino] = celda
            celdas_vecinas.append([vecino, nuevo_g, distance(vecino, goal), celda])

    if m.maze_map[celda]['N'] == 1:
        vecino = (x-1, y)
        nuevo_g = g + 1
        if vecino not in g_scores or nuevo_g < g_scores[vecino]:
            g_scores[vecino] = nuevo_g
            padres[vecino] = celda
            celdas_vecinas.append([vecino, nuevo_g, distance(vecino, goal), celda])
    
    if m.maze_map[celda]['W'] == 1:
        vecino = (x, y-1)
        nuevo_g = g + 1
        if vecino not in g_scores or nuevo_g < g_scores[vecino]:
            g_scores[vecino] = nuevo_g
            padres[vecino] = celda
            celdas_vecinas.append([vecino, nuevo_g, distance(vecino, goal), celda])

    if m.maze_map[celda]['E'] == 1:
        vecino = (x, y+1)
        nuevo_g = g + 1
        if vecino not in g_scores or nuevo_g < g_scores[vecino]:
            g_scores[vecino] = nuevo_g
            padres[vecino] = celda
            celdas_vecinas.append([vecino, nuevo_g, distance(vecino, goal), celda])

# Función para calcular la distáncia heurística entre dos celdas
def distance(cell1, cell2):
    x1, y1 = cell1
    x2, y2 = cell2
    return abs(x1 - x2) + abs(y1 - y2)

#Ordena por f_score (g + h) y retorna el mejor nodo
def seleccionar_mejor_nodo(celdas_vecinas):
    celdas_vecinas.sort(key=lambda x: x[1] + x[2])  # Ordenar por f = g + h
    return celdas_vecinas.pop(0)  # Retornar y eliminar el mejor

def aStar(m):
    # Debo añadir la celda vecina junto a su distancia recorrida y la minima faltante [(x,y),g_score,h_score,(x',y')]
    celdas_vecinas = []
    # Set para guardar las celdas que ya hemos visitado
    visitadas = set()
    # Set para guardar el
    g_scores = {start: 0}
    padres = {}

    # Inicializo el primero nodo en el punto de partida
    nodo_inicial = [start, 0, distance(start, goal), start]

    visitadas.add(nodo_inicial[0])

    # Llamo a la función expandir celda para ver las posibles celdas vecinas que hay junto al inicio
    expandir_celda(nodo_inicial, m, celdas_vecinas, g_scores, padres)

    # Hacemos un while para que siempre que haya una celda en la lista, mirar las celdas vecinas e intentar seguir avanzando
    while celdas_vecinas:
        # Seleccionamos la celda mas cercana con menor recorrido hecho para continuar desde ahí
        nodo_actual = seleccionar_mejor_nodo(celdas_vecinas)
        celda_actual = nodo_actual[0]

        # Si ya hemos llegado a la casilla de salida entonces formamos el camino
        if celda_actual == goal:
            forwardPath = {}
            celda = goal

            # Mientras la celda en la que estemos no sea la de inicio continuamos añadiendo celdas a la ruta
            while celda != start:
                padre = padres[celda]
                forwardPath[padre] = celda
                celda = padre

            return forwardPath
        
        # Si ya hemos visitado esta celda, la saltamos
        if celda_actual in visitadas:
            continue

        # Añadimos la celda visitada al set
        visitadas.add(celda_actual)
        expandir_celda(nodo_actual, m, celdas_vecinas, g_scores, padres)

pre_Astar = time.time()
path = aStar(m)
post_Astar = time.time()
print(post_Astar - pre_Astar)
a=maze.agent(m,footprints=True)
m.tracePath({a:path},delay=5)
m.run()