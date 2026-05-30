from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import pandas as pd

# Cargamos tu dataset actual
df = pd.read_csv("data/dataset_limpio.csv")

# Guardamos solo el texto de las reviews en un archivo de texto
with open("data/corpus.txt", "w", encoding="utf-8") as f:
    for review in df['review']:
        f.write(str(review) + "\n")

myTokenizer = Tokenizer(BPE(unk_token="[UNK]"))
myTokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
myTokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
files = [f"data/corpus.txt"]
myTokenizer.train(files, trainer)
myTokenizer.save("data/myTokenizer.json")