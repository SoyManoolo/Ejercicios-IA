palabra = "reconocer"

def palindromo(palabra: str):
    lista_palabra = []
    resultado = True

    for i in palabra: 
        lista_palabra.insert(len(lista_palabra), i)

    palabra_girada = lista_palabra[::-1]

    for i in range(len(lista_palabra)):
        if lista_palabra[i] != palabra_girada[i]: resultado = False

    return resultado

print(palindromo(palabra))