import pandas as pd
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import cv2

import common # mój moduł z powszechnymi funkcjami i stałymi

# ==========================================
# 1. ARCHITEKTURA CRNN (Z KONTROWANYM NAGŁÓWKIEM)
# ==========================================

class CRNNEncoder(nn.Module):
    """Główny trzon sieci wyciągający cechy (Backbone)."""
    def __init__(self, embedding_dim=128):
        super(CRNNEncoder, self).__init__()
        
        # Ekstrakcja cech z obrazu (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), # wys: 32 -> 16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), # wys: 16 -> 8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(16, 1))  # Sprowadza wysokość do 1 px
        )
        
        # Analiza sekwencji czasowej (BiLSTM)
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=64, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True
        )
        
        # Końcowy wektor cech (embedding)
        self.fc_embedding = nn.Linear(128, embedding_dim)

    def forward(self, x):
        features = self.cnn(x)                # [Batch, 128, 1, Szerokość']
        # print(features)
        features = features.squeeze(2)        # [Batch, 128, Szerokość']
        features = features.permute(0, 2, 1)  # [Batch, Szerokość', 128]
        
        lstm_out, _ = self.lstm(features)     # [Batch, Szerokość', 128]
        
        # Agregacja po czasie (uśrednienie po zmiennej szerokości)
        embedding = torch.mean(lstm_out, dim=1) # [Batch, 128]
        
        # Wektor wyjściowy stylistyki pisma
        output = self.fc_embedding(embedding)   # [Batch, embedding_dim]
        return output


class PretrainCRNNClassifier(nn.Module):
    """Pełny model z nagłówkiem klasyfikacyjnym używany TYLKO do pre-trainingu."""
    def __init__(self, num_classes, embedding_dim=128):
        super(PretrainCRNNClassifier, self).__init__()
        self.encoder = CRNNEncoder(embedding_dim=embedding_dim)
        # Klasyfikator rzutujący embedding na liczność autorów
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return logits, embedding

def collate_fn_pad(batch):
    """
    Funkcja łącząca próbki o RÓŻNYCH SZEROKOŚCIACH w jeden batch za pomocą paddingu.
    Gdy robimy batch > 1, obrazy wewnątrz batcha muszą mieć równy rozmiar.
    """
    images, labels = zip(*batch)
    
    # Znajdź maksymalną szerokość w danym batchu
    max_width = max(img.shape[2] for img in images)
    
    padded_images = []
    for img in images:
        pad_width = max_width - img.shape[-1]
        # Dopełniamy zerami z prawej strony: (pad_left, pad_right, pad_top, pad_bottom)
        padded_img = nn.functional.pad(img, (0, pad_width, 0, 0), value=0)
        padded_images.append(padded_img)
        
    return torch.stack(padded_images), torch.tensor(labels)

class HandwritingDataset(Dataset):
    """
    Sztuczny dataset do celów demonstracyjnych.
    Generuje obrazy o STAŁEJ WYSOKOŚCI (32px), ale ZMIENNEJ SZEROKOŚCI.
    """
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        author_id = self.df.iloc[idx]["user_class"]
        image_path = self.df.iloc[idx]["word_path"]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Nie można odczytać obrazu: {image_path}")

        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        image_tensor = image_tensor.unsqueeze(0)  # [1, Wysokość, Szerokość]

        return image_tensor, torch.tensor(int(author_id), dtype=torch.long)

class CachedHandwritingDataset(Dataset):
    def __init__(self, df):
        # Konwertujemy kolumny na natywne listy (błyskawiczny dostęp)
        self.labels = df["user_class"].astype(int).tolist()
        paths = df["word_path"].tolist()
        
        print("Wczytywanie całego zbioru do pamięci RAM...")
        self.images = []
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Błąd odczytu: {p}")
            # Normalizujemy i zamieniamy na tensor od razu
            tensor_img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
            self.images.append(tensor_img)
        print("Wczytano do pamięci RAM!")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Dostęp natychmiastowy z pamięci RAM (O(1))
        return self.images[idx], torch.tensor(self.labels[idx], dtype=torch.long)

def test_main():
    EMBEDDING_DIM = 128   # Rozmiar docelowego wektora cech
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 1e-3

    df_words = pd.read_parquet("df_words_preprocessed.parquet")
    
    classes = sorted(df_words["user_class"].unique())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    df_words["user_class"] = df_words["user_class"].map(class_to_idx)
    
    df_train, df_val, df_test = common.get_subsets(df_words, test_size=0.2, val_size=0.20)

    dataset = HandwritingDataset(df_train)
    
    generator = torch.Generator().manual_seed(common.GLOBAL_SEED)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator, collate_fn=collate_fn_pad)
    
    # Inicjalizacja modelu, funkcji straty i optymalizatora
    model = PretrainCRNNClassifier(num_classes=len(df_train["user_class"].unique()), embedding_dim=EMBEDDING_DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("--- ROZPOCZĘCIE ETAPU PRE-TRAININGU ---")
    model.train()
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Przepływ w przód
            logits, _ = model(images)
            loss = criterion(logits, labels)
            
            # Przepływ wsteczny
            loss.backward()
            optimizer.step()

            # Metryki
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100
        epoch_time = time.time() - epoch_start_time
        print(f"Epoka [{epoch+1}/{EPOCHS}] | Straty (Loss): {epoch_loss:.4f} | Dokładność (Acc): {epoch_acc:.2f}% | Czas: {epoch_time:.2f}s")

    print("\n--- ZAKOŃCZONO PRE-TRAINING ---")

    torch.save(model.encoder.state_dict(), "crnn_encoder_pretrained.pth")
    print("Zapisano wagi wyuczonego enkodera do pliku: 'crnn_encoder_pretrained.pth'")

    return model.encoder

if __name__ == "__main__":
    test_main()
