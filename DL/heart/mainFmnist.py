import torch as pt
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets
import torch.nn.functional as F

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=500, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=500, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.linear_relu_stack(x)
        return y
    
model = NeuralNetwork()
optimizer = pt.optim.SGD(model.parameters(), lr=0.05)
loss_error = nn.MSELoss()

def trainloop(dataloader: DataLoader, model: NeuralNetwork, optimizer: pt.optim.SGD, loss_error: nn.MSELoss):
    correct = 0

    for images, labels in dataloader:
        optimizer.zero_grad()
        y = model(images)
        labels_onehot = F.one_hot(labels, num_classes=10).float()
        loss = loss_error(y, labels_onehot)

        pred = y.argmax(1)
        correct = (pred == labels).sum().item()
        batch_accuracy = correct / len(labels)
        
        print(f"Loss: {loss.item():.4f}, Accuracy: {(100*batch_accuracy):.2f}%")

        loss.backward()
        optimizer.step()

epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    trainloop(train_dataloader, model, optimizer, loss_error)
    print(f"Training completed for epoch {epoch+1}")

pt.save(model.state_dict(), "model.pth")