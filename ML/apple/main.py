import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# Configuración de numpy para mejor visualización
np.set_printoptions(suppress=True, precision=3)

# Carga de datos numéricos ignorando cabecera y pie
datos_num = np.genfromtxt("./apple_quality.csv", delimiter=",", skip_header=1, skip_footer=1)
print("Vista de datos numéricos:\n", datos_num[:3]) # Imprimir solo primeras filas para no saturar

print("----------------------------------------------")

# Carga de datos como texto para extraer la columna de clases
datos_texto = np.genfromtxt("./apple_quality.csv", delimiter=",", dtype=str, skip_header=1, skip_footer=1)

# X representa las variables independientes, Y la variable dependiente (objetivo)
X = datos_num[:, :-1]
Y_labels = datos_texto[:, -1]

print("Primeras etiquetas:", Y_labels[:5])

# Convertimos las etiquetas de texto a valores binarios de forma vectorizada con numpy
Y_numerico = np.where(Y_labels == 'good', 1, 0)

# Division de datos usando sklearn
X_train, X_test, y_train, y_test = train_test_split(X, Y_numerico, test_size=0.3, random_state=42)

print(f"Set de entrenamiento (X): {X_train.shape}")
print(f"Set de prueba (X): {X_test.shape}")

# Inicializar y entrenar el modelo limitando su expansión
modelo_arbol = DecisionTreeClassifier(max_depth=4, random_state=42) 
modelo_arbol.fit(X_train, y_train)

# Predicción sobre los datos de test
pred = modelo_arbol.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"-> Precisión global del Árbol de Decisión: {acc:.4f}\n")

# Mostrar matriz de confusión
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=['Bad (0)', 'Good (1)'], cmap='Blues')
plt.show() # Añadido para que la matriz de confusión se dibuje en el script principal

nombres_caracteristicas = ["A_id", "Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"]

plt.figure(figsize=(40, 12))
plot_tree(modelo_arbol,
          feature_names=nombres_caracteristicas,
          class_names=["bad", "good"],
          filled=True,
          rounded=True,
          fontsize=10)

plt.title("Estructura del Árbol de Decisión", fontsize=20)
plt.show()
