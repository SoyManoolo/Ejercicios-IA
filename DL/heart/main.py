import torch as pt
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Definimos la clase que hará de red neuronal
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Indicamos la cantidad de capas que tendrá la red neuronal junto al tipo de funcion de activación
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    # Definimos la funcion forward donde dentro llamamos a las secuencias para calcular la y predicha
    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y

# Leemos el csv y lo guardamos en la variable df
df = pd.read_csv('./data/HeartDisease/heart.csv')

# Separamos los parametros de los resultados en dos variables
X = df.drop('target', axis=1).values
y = df['target'].values

# Normalizamos datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separamos los datos en entreno y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% para test
    random_state=42     # Para reproducibilidad
)

# Convertir a tensores
X_train_t = pt.FloatTensor(X_train)
y_train_t = pt.LongTensor(y_train)
X_test_t = pt.FloatTensor(X_test)
y_test_t = pt.LongTensor(y_test)

# Crear datasets
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

# Crear dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Definimos el modelo
model = NeuralNetwork()
optimizer = pt.optim.SGD(model.parameters(), lr=0.05)
loss_error = nn.MSELoss()


def trainingloop(train_dataloader: DataLoader, model: NeuralNetwork, optimizer: pt.optim.SGD, loss_error: nn.MSELoss):
    # Contador para saber la cantidad de aciertos del modelo mientras se entrena
    correct = 0
    for features, labels in train_dataloader:
        # Dejamos los gradientes a 0
        optimizer.zero_grad()
        # Llamamos al modelo para calcular la y predicha
        y = model(features)

        # Hacemos onehot del label para poder hacer la funcion de perdida (error)
        labels_onehot = F.one_hot(labels, num_classes=2).float()
        # Calculamos el error que tiene el modelo en esta iteracion
        loss = loss_error(y, labels_onehot)

        pred = y.argmax(1)
        correct = (pred == labels).sum().item()
        batch_accuracy = correct / len(labels)
        
        print(f"Loss: {loss.item():.4f}, Accuracy: {(100*batch_accuracy):.2f}%")

        # Calculamos los gradientes
        loss.backward()
        # Modificamos los pesos y bias
        optimizer.step()

epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    trainingloop(train_loader, model, optimizer, loss_error)
    print(f"Training completed for epoch {epoch+1}")

