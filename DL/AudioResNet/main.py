import os

# --- Librerías de datos ---
import numpy as np           # Operaciones numéricas con arrays (vectores y matrices)
import pandas as pd          # Manipulación de datos en formato tabla (DataFrames)

# --- Librerías de audio ---
import librosa               # Análisis y procesamiento de audio

# --- Librerías de visualización ---
import matplotlib.pyplot as plt  # Crear gráficos

# --- PyTorch: Framework de Deep Learning ---
import torch as pt                      # Librería principal de PyTorch
import torch.nn as nn                   # Módulos de redes neuronales (capas, funciones)
import torch.optim as optim             # Optimizadores (SGD, Adam, etc.)
from torch.utils.data import Dataset, DataLoader  # Clases para manejar datos

# --- TorchVision: Modelos y transformaciones de imágenes ---
import torchvision.models as models     # Modelos pre-entrenados (ResNet, VGG, etc.)
import torchvision.transforms as transforms  # Transformaciones de imágenes

# --- Utilidades ---
from tqdm import tqdm                   # Barras de progreso
from sklearn.metrics import confusion_matrix, classification_report  # Métricas

if pt.cuda.is_available():
    pt.cuda.empty_cache()

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

# Configuración de directorios

# Directorio base de los datos
DATASET_PATH = "data/ESC-50-master"
# Directorio donde se encuentran los audios
AUDIO_DIR = os.path.join(DATASET_PATH, "audio")
SPECTROGRAM_DIR = os.path.join(DATASET_PATH, "audio_data")
# Directorio donde se encuentra el csv 
META_PATH = os.path.join(DATASET_PATH, "meta", "esc50.csv")

BATCH_SIZE = 16
NUM_EPOCAS = 15
NUM_CLASES = 50
LR = 0.001

# modelo
model = models.resnet50(weights="IMAGENET1K_V2")

# Añadimos clasificación final del modelo, ya que solo disponemos de 50 clases
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, NUM_CLASES)
)

audio_cfg = dict(
    n_mels = 224,        # Bandas de frecuencia Mel
    hop_length = 256,    # Salto entre frames
    win_length = 2048,    # Tamaño de ventana (diferente al anterior)
    n_fft = 2048,        # Tamaño FFT
    fmax = None         # Frecuencia máxima (22 kHz)
)

# Dataset
class ESC50Dataset(Dataset):
    def __init__(self, dataframe, spectrogram_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.spectrogram_dir = spectrogram_dir
        self.transform = transform
        self.categorias = sorted(dataframe['category'].unique())
        self.cat_to_idx = {cat: idx for idx, cat in enumerate(self.categorias)}
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        fila = self.dataframe.iloc[idx]
        etiqueta = self.cat_to_idx[fila['category']]
        
        # Cargar espectrograma pre-computado
        spec_path = os.path.join(self.spectrogram_dir, fila['filename'].replace('.wav', '.npy'))
        mel_spec = np.load(spec_path)
        
        # Normalizar [0, 1] y convertir a tensor RGB
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
        mel_spec = pt.tensor(mel_spec, dtype=pt.float32).unsqueeze(0).repeat(3, 1, 1)
        
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        return mel_spec, etiqueta
    
# Preparación de los datos

# Leer csv
df = pd.read_csv(META_PATH)
# Separamos los features de los labels
df_train = df[df['fold'].isin([1, 2, 3, 4])]
df_val = df[df['fold'] == 5]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Definimos los dataset de entreno y de evaluacion
train_dataset = ESC50Dataset(df_train, SPECTROGRAM_DIR, transform)
val_dataset = ESC50Dataset(df_val, SPECTROGRAM_DIR, transform)

# Definimos los dataloader de entreno y de evaluacion
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Función para pasar el audio a espectrograma
def Audio2Spectrogram(audio_file, dict_cfg):
    audio, Fs = librosa.load(audio_file)           # 1. Cargar audio
    spectrogram = librosa.feature.melspectrogram(  # 2. Calcular Mel-espectrograma
        y=audio, sr=Fs, **dict_cfg
    )
    spectrogram = librosa.power_to_db(spectrogram) # 3. Convertir a dB
    return librosa.util.normalize(spectrogram)     # 4. Normalizar a [-1, 1]

# Generación de espectrogramas
for filename in tqdm(df['filename']):
    save_path = os.path.join(SPECTROGRAM_DIR, filename.replace('.wav', '.npy'))
    if not os.path.exists(save_path):
        AUDIO_FILE_PATH = os.path.join(AUDIO_DIR, filename)
        spec = Audio2Spectrogram(AUDIO_FILE_PATH, audio_cfg)
        np.save(save_path, spec)

# Configuración de entrenamiento
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

def train (model: models.ResNet, loss_error: nn.CrossEntropyLoss, train_dataloader: DataLoader, optimizer: pt.optim.Adam):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for features, labels in train_dataloader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()

        y = model(features)

        loss = loss_error(y, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        correct += (y.argmax(1) == labels).sum().item()
        total += labels.size(0)

        batch_accuracy = correct / len(labels)

        print(f"Loss: {loss.item():.4f}, Accuracy: {(100*batch_accuracy):.2f}%")

    return total_loss / total, correct / total

def eval(model, loss, dataloader):
    return

for epoch in range(NUM_EPOCAS):
    train_loss, train_acc = train(model, loss, train_loader, optimizer)
    print(f"Ep {epoch+1:2d}/{NUM_EPOCAS} | Train: {train_acc*100:5.2f}%")