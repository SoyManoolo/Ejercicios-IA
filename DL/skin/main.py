import torch as pt
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F

# Definimos el tamaño de las imagenes para que todas sean iguales
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Primero cargamos todo el dataset
dataset = datasets.ImageFolder(root='./data', transform=transform)

# Definimos el tamaño de los dataset de entreno y test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Dividimos aleatoriamente los datos entre el dataset de entreno y el de test
train_dataset, test_dataset = pt.utils.data.random_split(
    dataset, 
    [train_size, test_size]
)

# Creamos los dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class NeuralNetwork (nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(100352, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y

model = NeuralNetwork()
optimizer = pt.optim.SGD(model.parameters(), lr=0.005)
loss_error = nn.CrossEntropyLoss()

def train_loop(train_dataloader: DataLoader, model: NeuralNetwork, optimizer: pt.optim.SGD, loss: nn.CrossEntropyLoss):
    correct = 0
    for features, labels in train_dataloader:
        optimizer.zero_grad()

        y = model(features)

        loss = loss_error(y, labels)

        pred = y.argmax(1)
        correct = (pred == labels).sum().item()
        batch_accuracy = correct / len(labels)
        
        print(f"Loss: {loss.item():.4f}, Accuracy: {(100*batch_accuracy):.2f}%")

        loss.backward()
        optimizer.step()


epochs = 15
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_loop(train_loader, model, optimizer, loss_error)
    print(f"Training completed for epoch {epoch+1}")

pt.save(model.state_dict(), 'skin_cancer_model.pth')