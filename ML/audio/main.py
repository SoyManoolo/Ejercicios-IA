import sklearn as sk
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from numpy.fft import fft

# Cargamos el json
with open("audioIA.json") as f:
    data = json.load(f)

Fs, audio = sio.wavfile.read("audioIA.wav")
X = []
y = []

for sample in data:
    start = float(sample["start"])
    end = float(sample["end"])
    vocal = sample["vocal"]

    # Cortamos el audio
    cut = audio[int(start*Fs):int(end*Fs), 0]

    # Extraer características (ejemplo: usar tu código de FFT)
    fourier = np.fft.fft(cut)
    Fsmall = fourier[0:300]
    toprocess = np.sqrt((np.real(Fsmall)**2 + np.imag(Fsmall)**2))

    filter_size = 5
    out = np.zeros(len(toprocess) - filter_size)
    for i in range(len(toprocess) - filter_size):
        out[i] = np.sum(toprocess[i:i+filter_size])
    out[0:15] = 0

    # Características: posición del máximo, valor máximo, media, etc.
    features = [
        np.argmax(out),           # Posición del pico
        np.max(out),              # Valor máximo
        np.mean(out),             # Media
        np.std(out),              # Desviación estándar
        # Puedes agregar más características
    ]
    
    X.append(features)
    y.append(vocal)

X = np.array(X)
y = np.array(y)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Entrenar SVM
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"\nAccuracy en entrenamiento: {train_score:.2f}")
print(f"Accuracy en test: {test_score:.2f}")

# ===== VISUALIZACIÓN DE LÍNEAS DE SEPARACIÓN =====

# Usar solo las primeras 2 características para visualización 2D
X_2d = X[:, :2]  # Posición del pico y valor máximo

# Mapear vocales a números para el colormap
vocal_to_num = {vocal: i for i, vocal in enumerate(np.unique(y))}
y_numeric = np.array([vocal_to_num[vocal] for vocal in y])

# Re-entrenar el modelo con solo 2 características
model_2d = SVC(kernel='rbf', C=1.0, gamma='scale')
model_2d.fit(X_2d, y)

# Crear una malla para visualizar las regiones de decisión
# Usar linspace para crear una malla con un número fijo de puntos
x_min, x_max = X_2d[:, 0].min() - 10, X_2d[:, 0].max() + 10
y_min, y_max = X_2d[:, 1].min() - 1e5, X_2d[:, 1].max() + 1e5

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predecir para cada punto de la malla
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])

# Convertir las predicciones de vocales a números
Z_numeric = np.array([vocal_to_num[vocal] for vocal in Z])
Z_numeric = Z_numeric.reshape(xx.shape)

# Crear el plot
plt.figure(figsize=(12, 10))

# Plotear las regiones de decisión (usar Z_numeric en lugar de Z)
plt.contourf(xx, yy, Z_numeric, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.contour(xx, yy, Z_numeric, colors='k', linewidths=0.8, alpha=0.6)

# Plotear los puntos de entrenamiento
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_numeric, 
                     cmap=plt.cm.RdYlBu, edgecolors='black', 
                     s=100, alpha=0.8)

# Plotear los vectores de soporte
plt.scatter(X_2d[model_2d.support_, 0], 
           X_2d[model_2d.support_, 1],
           s=250, linewidth=2, facecolors='none', 
           edgecolors='green', label='Support Vectors')

# Añadir etiquetas de vocales en los puntos
for i, vocal in enumerate(y):
    plt.annotate(vocal, (X_2d[i, 0], X_2d[i, 1]), 
                fontsize=8, alpha=0.7, ha='center')

plt.xlabel('Posición del Pico (argmax)', fontsize=12)
plt.ylabel('Valor Máximo', fontsize=12)
plt.title('SVM - Clasificación de Vocales\nLíneas de Separación', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Crear colorbar con nombres de vocales
cbar = plt.colorbar(scatter, ticks=range(len(vocal_to_num)))
cbar.set_label('Vocal', fontsize=12)
cbar.ax.set_yticklabels(list(vocal_to_num.keys()))

print(f"\nAccuracy del modelo 2D: {model_2d.score(X_2d, y):.2f}")

plt.tight_layout()
plt.show()