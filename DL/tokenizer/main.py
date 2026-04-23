from tokenizers import Tokenizer
import torch as pt
import torch.nn as nn
from torch.nn import Embedding
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import random_split

# Digo donde está el archivo tokenizer
tokenizer: Tokenizer = Tokenizer.from_file("data/myTokenizer.json")

# Digo el id del pad para las frases que no lleguen a la cantidad deseada (128)
tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", length=128)
tokenizer.enable_truncation(max_length=128)

# Funcion para pasar la frase a tokens a partir del tokenizer
def phraseToTensor(list):
    encodings = tokenizer.encode_batch(list)
    ids_list = [e.ids for e in encodings]
    return pt.tensor(ids_list).long()

VOCAB_SIZE = 30522  # Definido en tu BpeTrainer
PAD_TOKEN_ID = 3    # El ID de [PAD] en tu myTokenizer.json
EMBED_DIM = 128     # Eliges tú según la potencia que necesites

df = pd.read_csv("./data/dataset_limpio.csv")
labels_num = [1 if s == 'positive' else 0 for s in df['sentiment']]
X = phraseToTensor(df['review'].tolist())
Y = pt.tensor(labels_num).long()

# Aqui inicializo el dataset 
dataset = TensorDataset(X, Y)
train_size = int(0.8 * len(dataset)) # 800 para entrenar
test_size = len(dataset) - train_size # 200 para test

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Ahora creas dos loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class MiModelo(nn.Module):
    def __init__(self):
        super().__init__()

        # num_embeddings es el numero de vocab_size que se eligió en el tokenizer
        # embedding_dim es el tamaño del vector denso que definirá cada palabra
        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=EMBED_DIM,
            padding_idx=PAD_TOKEN_ID
        )
        self.lstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=256,
            batch_first=True
        )
        self.fc = nn.Linear(256,2)
    
    def forward(self, x):
        x = self.embedding(x)
        outpt, (hn, cn) = self.lstm(x)
        resumen = hn[-1]
        return self.fc(resumen)

model = MiModelo()
loss = nn.CrossEntropyLoss()
lr = 0.003
optimizer = pt.optim.Adam(model.parameters(), lr)

# Funcion para entrenar el embedding
def trainloop(model: MiModelo, loss_fn: nn.CrossEntropyLoss, optimizer: pt.optim.Adam, dataloader: DataLoader):
    model.train()
    for epoch in range(7):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            y = model(batch_x)
            loss = loss_fn(y, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Cálculo de Accuracy
            _, predicted = pt.max(y, 1) # Obtiene el índice de la mayor probabilidad
            correct_predictions += (predicted == batch_y).sum().item()
            total_samples += batch_y.size(0)

        epoch_loss = total_loss / len(dataloader)
        epoch_acc = correct_predictions / total_samples
        print(f"Época {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

def testloop():
    return

trainloop(model, loss, optimizer, train_loader)