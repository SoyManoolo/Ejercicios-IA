import torch as pt
from torch import nn

class BloqueRes (nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv_stack(x)
        out += identity
        out = self.relu(out)

        return out

class NeuralNetwork (nn.Module):
    def __init__(self):
        super().__init__()

        self.capa_inicial = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.bloque1 = BloqueRes()
        self.bloque2 = BloqueRes()

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        x = self.capa_inicial(x)
        x = self.bloque1(x)
        x = self.bloque2(x)
        x = self.global_avg_pool(x)
        x = pt.flatten(x,1)
        return self.classifier(x)