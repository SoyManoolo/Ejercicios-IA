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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(PositionalEncoding, self).__init__()
        

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
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(256, 2)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x)
    
model = MiModelo()
loss = nn.CrossEntropyLoss()
lr = 0.0005
optimizer = pt.optim.Adam(model.parameters(), lr)

def train_loop(model: MiModelo, loss_fn: nn.CrossEntropyLoss, optimizer: pt.optim.Adam, dataloader: DataLoader):
    model.train()
    for epoch in range(10):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        continue
    return