import csv
import re

def limpiar_csv(archivo_entrada, archivo_salida):
    regex_html = re.compile('<.*?>')
    
    with open(archivo_entrada, mode='r', encoding='utf-8') as f_in, \
         open(archivo_salida, mode='w', encoding='utf-8') as f_out:
        
        reader = csv.DictReader(f_in)

        for row in reader:
            # Limpiamos el texto
            texto_limpio = re.sub(regex_html, ' ', row['review'])
            # Escribimos solo el texto en una línea nueva
            f_out.write(texto_limpio.strip() + "\n")

    print(f"Dataset limpio guardado en: {archivo_salida}")

# Úsalo así:
limpiar_csv("./data/dataset.csv", "./data/dataset_limpio.txt")