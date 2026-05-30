import csv
import re

def limpiar_y_guardar_csv(archivo_entrada, archivo_salida):
    regex_html = re.compile('<.*?>')
    
    with open(archivo_entrada, mode='r', encoding='utf-8') as f_in, \
         open(archivo_salida, mode='w', encoding='utf-8', newline='') as f_out:
        
        reader = csv.DictReader(f_in)
        # Definimos las columnas que tendrá el nuevo archivo
        columnas = ['review', 'sentiment']
        writer = csv.DictWriter(f_out, fieldnames=columnas)
        
        # Escribimos la cabecera (review, sentiment)
        writer.writeheader()

        for row in reader:
            # 1. Limpiamos el texto
            texto_limpio = re.sub(regex_html, ' ', row['review']).strip()
            
            # 2. Escribimos la fila completa con el texto limpio y su etiqueta original
            writer.writerow({
                'review': texto_limpio,
                'sentiment': row['sentiment']
            })

    print(f"Nuevo CSV guardado con éxito en: {archivo_salida}")

# Úsalo así:
limpiar_y_guardar_csv("./data/dataset.csv", "./data/dataset_limpio.csv")