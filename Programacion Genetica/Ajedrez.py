import random

# Utilitzant programació genetica, crea un algorisme que trobi una possible solució del problema de les 8 reinas generalitzada per taules d’escacs de NxN.

mutacion = 10
gen_actual = 75
gen_ant = 25

# Funcion para generar individuos - LISTO
def init_individu(mida_tauler):
    return random.sample(range(mida_tauler), mida_tauler)

# Funcion para iniciar la poblacion con la medida del tablero - LISTO
def init_population(mida_tauler, mida_pop):
    poblacion_actual = []
    poblacion_anterior = []

    for individu in range(mida_pop):
        poblacion_actual.append(init_individu(mida_tauler))

    print(poblacion_actual)
    fitness(mida_tauler, poblacion_actual)

# Funcion para mezclar padres y generar hijos - FALTA MUTACIÓN
def crossover(individu1, individu2):
    new_ind = []
    mitad = len(individu1) // 2

    for i in range(mitad):
        new_ind.append(individu1[i])

    for i in individu2:
        if i not in new_ind:
            new_ind.append(i)

    return new_ind

# Funcion para calcular el fitness de cada individuo - LISTO
def fitness(mida_tauler, pop):
    fitness_actual = []
    fitness_anterior = []

    for individu in pop:
        choques = 0
        for i in range(mida_tauler):
            for j in range(i + 1, mida_tauler):
                if abs(individu[i] - individu[j]) == abs(i - j):
                    choques += 1
        fitness_actual.append(choques)

    print(fitness_actual)
    return

# Funcion para elegir los individuos que permanecerán en la siguiente generacion
def selection(fit_actual):
    for i in fit_actual:
        print(i)
    return

init_population(8, 100)