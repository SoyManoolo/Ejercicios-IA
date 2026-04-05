from tokenizers import Tokenizer
import torch as pt
import torch.nn as nn
from torch.nn import Embedding

# Digo donde está el archivo tokenizer
tokenizer: Tokenizer = Tokenizer.from_file("data/myTokenizer.json")

tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", length=128)
tokenizer.enable_truncation(max_length=128)

def phraseToToken(list):
    encodings = tokenizer.encode_batch(list)
    ids_list = [e.ids for e in encodings]
    return pt.tensor(ids_list).long()

VOCAB_SIZE = 30522  # Definido en tu BpeTrainer
PAD_TOKEN_ID = 3    # El ID de [PAD] en tu myTokenizer.json
EMBED_DIM = 256     # Eliges tú según la potencia que necesites


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
            input_size=0,
            hidden_size=0,
            batch_first=True
        )
        self.fc = nn.Linear(512,2)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        return self.fc(x)


def tokenToTensor():
    return

def trainloop():
    return

def testloop():
    return

