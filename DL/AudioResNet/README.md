
# n_mels: Número de bandas de frecuencia Mel
# - Determina la altura (resolución frecuencial) del espectrograma
# - Más bandas = más detalle en frecuencias

# hop_length: Número de muestras entre frames consecutivos
# - Determina la resolución temporal (ancho del espectrograma)
# - Menor valor = más frames = mejor resolución temporal

# win_length: Tamaño de la ventana de análisis en muestras
# - Debe ser menor o igual que n_fft
# - Ventana más grande = mejor resolución frecuencial pero peor temporal

# n_fft: Tamaño de la ventana FFT
# - Siempre potencia de 2 para eficiencia
# - Mayor = mejor resolución frecuencial

# fmax: Frecuencia máxima a considerar (en Hz)
# - El oído humano percibe hasta ~20000 Hz
# Ponemos None para que solamente muestre hasta la frecuencia máxima del audio