from typing import List

co_data = ["first", 12, 47, 0, -4, 23.02, [2, 3, 4, 8, -3], "last", 3, 100, -31]
lista_limpia = []
lista_ordenada = []
numero_pequenyo = 0

def clean_order(lista: List):
	for i in lista:
		if type(i) == int:
			lista_limpia.insert(len(lista_limpia), i)


	while len(lista_limpia) > 0:
		numero_pequenyo = lista_limpia[0]
  
		for j in range(len(lista_limpia)):
			if numero_pequenyo > lista_limpia[j]:
				numero_pequenyo = lista_limpia[j]
      
		lista_ordenada.insert(len(lista_ordenada), numero_pequenyo)
		lista_limpia.remove(numero_pequenyo)

	return lista_ordenada

print(clean_order(co_data))