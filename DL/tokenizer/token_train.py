from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


myTokenizer = Tokenizer(BPE(unk_token="[UNK]"))
myTokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
myTokenizer.pre_tokenizer = Whitespace()


trainer = BpeTrainer(vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
files = [f"data/dataset_limpio.txt" for split in ["test", "train", "valid"]] # Me falta seleccionar un dataset, descargarlo y pasarlo a files para que el tokenizer se pueda "entrenar" y generar el archivo .json
myTokenizer.train(files, trainer)
myTokenizer.save("data/myTokenizer.json")