import random

# Utilitzant programació genetica, crea un algorisme que trobi una possible solució del problema de les 8 reinas generalitzada per taules d’escacs de NxN.

# Mutacion = 20%

# Reparticion de poblacion entre cada generacion:
# Generacion actual = 75%
# Generacion anterior = 25%

# Mejores fit = 65%
# Peores fit = 35%

# Funcion para generar individuos - LISTO
def init_individu(mida_tauler):
    return random.sample(range(mida_tauler), mida_tauler)

# Funcion para iniciar la poblacion con la medida del tablero - LISTO
def init_population(mida_tauler, mida_pop):
    poblacion_actual = []

    for i in range(mida_pop):
        poblacion_actual.append(init_individu(mida_tauler))

    print(poblacion_actual)

    return poblacion_actual

# Funcion para mezclar padres y generar hijos - LISTO
def crossover(individu1, individu2):
    new_ind = []
    mitad = len(individu1) // 2

    for i in range(mitad):
        new_ind.append(individu1[i])

    for i in individu2:
        if i not in new_ind:
            new_ind.append(i)

    if random.randint(1, 100) <= 20:
        i, j = random.sample(range(len(new_ind)), 2)
        new_ind[i], new_ind[j] = new_ind[j], new_ind[i]
        print(f"Individuo mutado: {new_ind}")

    return new_ind

# Funcion para calcular el fitness de cada individuo - LISTO
def fitness(mida_tauler, poblacion_actual):
    fitness_actual = []

    for individu in poblacion_actual:
        choques = 0
        for i in range(mida_tauler):
            for j in range(i + 1, mida_tauler):
                if abs(individu[i] - individu[j]) == abs(i - j):
                    choques += 1
        fitness_actual.append(choques)

        if choques == 0:
            print(f"Este es uno de los posibles resultados: {individu}")
            return fitness_actual, True
    
    print(fitness_actual)
    return fitness_actual, False

# Funcion para elegir los individuos que permanecerán en la siguiente generacion
def selection(pop_actual, fit_actual, pop_anterior, fit_anterior):
    new_poblation = []

    total_actual = int(len(pop_actual) * 0.75)
    total_anterior = len(pop_actual) - total_actual

    mejores_actual = int(total_actual * 0.65)
    peores_actual = total_actual - mejores_actual

    mejores_anterior = int(total_anterior * 0.65)
    peores_anterior = total_anterior - mejores_anterior

    fit_actual_copy = fit_actual[:]
    fit_anterior_copy = fit_anterior[:]

    for _ in range(mejores_actual):
        mejor_fitness = min(fit_actual_copy)
        mejor_indice = fit_actual_copy.index(mejor_fitness)
        new_poblation.append(pop_actual[mejor_indice])
        fit_actual_copy[mejor_indice] = 999999
    
    for _ in range(peores_actual):
        indice = random.randint(0, len(pop_actual) - 1)
        new_poblation.append(pop_actual[indice])
    
    for _ in range(mejores_anterior):
        mejor_fitness = min(fit_anterior_copy)
        mejor_indice = fit_anterior_copy.index(mejor_fitness)
        new_poblation.append(pop_anterior[mejor_indice])
        fit_anterior_copy[mejor_indice] = 999999
    
    for _ in range(peores_anterior):
        indice = random.randint(0, len(pop_anterior) - 1)
        new_poblation.append(pop_anterior[indice])

    return new_poblation

# Funcion main que ejecuta todas las funciones
def main_algorithm(mida_tauler, mida_pop):
    poblacion_actual = init_population(mida_tauler, mida_pop)
    poblacion_anterior = []

    fitness_actual, resuelto = fitness(mida_tauler, poblacion_actual)
    if resuelto: return

    fitness_anterior = []

    while resuelto == False:

        nueva_generacion = []
        for _ in range(mida_pop):
            padre1 = random.choice(poblacion_actual)
            padre2 = random.choice(poblacion_actual)
            nueva_generacion.append(crossover(padre1, padre2))
        print(nueva_generacion)

        poblacion_anterior = poblacion_actual
        poblacion_actual = nueva_generacion

        fitness_anterior = fitness_actual
        fitness_actual, resuelto = fitness(mida_tauler, nueva_generacion)

        poblacion_actual = selection(poblacion_actual, fitness_actual, poblacion_anterior, fitness_anterior)

main_algorithm(30, 500)