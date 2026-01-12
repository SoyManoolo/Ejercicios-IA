import torch as pt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from main import NeuralNetwork

# Definir las mismas transformaciones que en entrenamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Cargar el dataset completo
dataset = datasets.ImageFolder(root='./data', transform=transform)

# Dividir igual que en entrenamiento
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = pt.utils.data.random_split(dataset, [train_size, test_size])

# Crear el DataLoader de test
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Crear modelo y cargar pesos guardados
model = NeuralNetwork()
model.load_state_dict(pt.load('skin_cancer_model.pth'))
model.eval()

# Función de evaluación
loss_error = nn.CrossEntropyLoss()

def test_loop(test_dataloader: DataLoader, model: NeuralNetwork):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    with pt.no_grad():
        for features, labels in test_dataloader:
            y = model(features)
            loss = loss_error(y, labels)
            
            pred = y.argmax(1)
            correct += (pred == labels).sum().item()
            total += len(labels)
            total_loss += loss.item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_dataloader)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

# Ejecutar evaluación
test_loop(test_loader, model)