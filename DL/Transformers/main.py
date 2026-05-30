from tokenizers import Tokenizer
import torch as pt
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import random_split
import math
from tqdm import tqdm

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128, dropout: float =0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = pt.zeros(max_len, d_model)
        position = pt.arange(0, max_len, dtype=pt.float).unsqueeze(1)
        div = pt.exp(-1 * (pt.arange(0, d_model, 2) / d_model) * math.log(10000.0))
        pe[:, 0::2] = pt.sin(position * div)
        pe[:, 1::2] = pt.cos(position * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)].unsqueeze(1)
        return self.dropout(x)

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

        self.pos_encoding = PositionalEncoding(d_model=EMBED_DIM)
        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=8)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(EMBED_DIM, 2)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.fc(x)
    
model = MiModelo()
loss = nn.CrossEntropyLoss()
lr = 0.0005
optimizer = pt.optim.Adam(model.parameters(), lr)

def train_loop(model: MiModelo, loss_fn: nn.CrossEntropyLoss, optimizer: pt.optim.Adam, dataloader: DataLoader):
    for epoch in range(5):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        # Añadimos tqdm al dataloader para ver la barrita
        progress_bar = tqdm(dataloader, desc=f"Época {epoch+1}")
        
        for batch_x, batch_y in progress_bar:
            optimizer.zero_grad()
            y = model(batch_x)
            loss = loss_fn(y, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = pt.max(y, 1)
            correct_predictions += (predicted == batch_y).sum().item()
            total_samples += batch_y.size(0)
            
            # Actualiza la info en la barra de progreso
            progress_bar.set_postfix(loss=loss.item(), acc=correct_predictions/total_samples)

        print(f"Época {epoch+1} Terminada | Acc: {correct_predictions/total_samples:.4f}")
        
train_loop(model, loss, optimizer, train_loader)