import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import common # mój moduł z powszechnymi funkcjami i stałymi

from sklearn.model_selection import train_test_split


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
            nn.MaxPool2d(kernel_size=(8, 1))  # Sprowadza wysokość do 1 px
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


# ==========================================
# 2. PRZYKŁADOWY DATASET I BATCHING
# ==========================================

class SyntheticHandwritingDataset(Dataset):
    """
    Sztuczny dataset do celów demonstracyjnych.
    Generuje obrazy o STAŁEJ WYSOKOŚCI (32px), ale ZMIENNEJ SZEROKOŚCI.
    """
    def __init__(self, num_samples=500, num_classes=10):
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Losujemy szerokość obrazu od 100px do 300px dla każdego słowa
        width = torch.randint(100, 301, (1,)).item()
        # Generujemy losowy obraz słowa [1, 32, width]
        image = torch.randn(1, 32, width)
        # Identyfikator autora (0 do num_classes - 1)
        author_id = torch.randint(0, self.num_classes, (1,)).squeeze(0)
        return image, author_id

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
        pad_width = max_width - img.shape[2]
        # Dopełniamy zerami z prawej strony: (pad_left, pad_right, pad_top, pad_bottom)
        padded_img = nn.functional.pad(img, (0, pad_width, 0, 0), value=0)
        padded_images.append(padded_img)
        
    return torch.stack(padded_images), torch.tensor(labels)


# ==========================================
# 3. PĘTLA TRENINGOWA (PRE-TRAINING)
# ==========================================

def pretrain_model():
    NUM_CLASSES = 10      # Liczba autorów w zbiorze treningowym
    EMBEDDING_DIM = 128   # Rozmiar docelowego wektora cech
    BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 1e-3

    # Przygotowanie danych
    dataset = SyntheticHandwritingDataset(num_samples=200, num_classes=NUM_CLASSES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad)

    # Inicjalizacja modelu, funkcji straty i optymalizatora
    model = PretrainCRNNClassifier(num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("--- ROZPOCZĘCIE ETAPU PRE-TRAININGU ---")
    model.train()
    
    for epoch in range(EPOCHS):
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
        print(f"Epoka [{epoch+1}/{EPOCHS}] | Straty (Loss): {epoch_loss:.4f} | Dokładność (Acc): {epoch_acc:.2f}%")

    print("\n--- ZAKOŃCZONO PRE-TRAINING ---")

    # ==========================================
    # 4. ZAPIS WAG DLA SIECI SYJAMSKIEJ
    # ==========================================
    
    # Wariant A: Zapisujemy tylko wyuczoną sekcję ENKODERA (CNN + BiLSTM)
    torch.save(model.encoder.state_dict(), "crnn_encoder_pretrained.pth")
    print("Zapisano wagi wyuczonego enkodera do pliku: 'crnn_encoder_pretrained.pth'")

    return model.encoder


class HandwritingCRNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super(HandwritingCRNN, self).__init__()
        
        # 1. Warstwy splotowe (CNN)
        # Wejście: [Batch, 1, 32, Szerokość]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), # Zmniejsza wys. do 16, szer. o połowę
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), # Zmniejsza wys. do 8, szer. o połowę
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(8, 1))  # Sprowadza wysokość do 1, szerokość bez zmian
        )
        
        # 2. Warstwa Rekurencyjna (BiLSTM)
        # Wejście do LSTM będzie miało rozmiar cech = 128 (liczba kanałów z CNN)
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=64, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True
        )
        
        # Ponieważ LSTM jest dwukierunkowy (BiLSTM), jego wyjście ma rozmiar: hidden_size * 2 = 128
        # Ostateczna warstwa liniowa mapująca na pożądany wymiar embeddingu
        self.fc = nn.Linear(128, embedding_dim)

    def forward_once(self, x):
        # Krok 1: Ekstrakcja cech przez CNN
        # Wyjście z cnn: [Batch, Kanały(128), Wysokość(1), Szerokość']
        features = self.cnn(x)
        
        # Krok 2: Map-to-Sequence (Przygotowanie dla RNN)
        # Pozbywamy się wymiaru wysokości (który wynosi 1)
        features = features.squeeze(2) # Kształt: [Batch, Kanały(128), Szerokość']
        
        # Zamieniamy miejscami wymiary, aby szerokość (kroki czasowe) była na drugim miejscu
        # LSTM w PyTorch (z batch_first=True) oczekuje: [Batch, Kroki_Czasowe, Cechy]
        features = features.permute(0, 2, 1) # Kształt: [Batch, Szerokość', Kanały(128)]
        
        # Krok 3: Przetwarzanie przez BiLSTM
        # wyjście lstm_out: [Batch, Szerokość', 128]
        lstm_out, _ = self.lstm(features)
        
        # Krok 4: Agregacja po czasie (Global Average Pooling po osi czasu)
        # Uśredniamy wszystkie kroki czasowe (oś szerokości), niezależnie od tego czy było ich 32, czy 64
        embedding = torch.mean(lstm_out, dim=1) # Kształt: [Batch, 128]
        
        # Krok 5: Rzutowanie na ostateczny wymiar
        output = self.fc(embedding) # Kształt: [Batch, embedding_dim]
        return output

    def forward(self, input1, input2):
        # Sieć syjamska przetwarza oba obrazy tym samym modelem
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

def get_subsets(df, test_size=0.2, val_size=0.20, random_state=common.GLOBAL_SEED):
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1.0")
    
    relative_val_size = val_size / (1 - test_size)

    # Zbiór treningowy + walidacyjny
    df_train_val, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['user_class']
    )
    
    # Podziel zbiór treningowo-walidacyjny na treningowy (75%) i walidacyjny (25%)
    df_train, df_val = train_test_split(
        df_train_val, test_size=relative_val_size, random_state=random_state, stratify=df_train_val['user_class']
    )
    
    return df_train, df_val, df_test

# --- TESTOWANIE DZIAŁANIA I RÓŻNYCH ROZMIARÓW ---

def main():
    # Zaimportuj zbiór danych oraz przygotuj podzbiory
    df = pd.read_parquet("df.parquet")
    df_words = common.build_word_dataframe(df, pics_path="dataset_words/words")

    df_train, df_val, df_test = get_subsets(df_words, test_size=0.2, val_size=0.20)
    
    # Inicjalizacja modelu
    model = HandwritingCRNN(embedding_dim=128)
    model.eval() # Tryb ewaluacji (wyłącza dropout itp.)

    print("--- TEST PRÓBKI 1 (Krótkie słowo: 32x128) ---")
    # Tworzymy sztuczny obraz (Batch=1, Kanał=1, Wysokość=32, Szerokość=128)
    probka_krotka = torch.randn(1, 1, 32, 128)
    emb1 = model.forward_once(probka_krotka)
    print(f"Rozmiar wejściowy: {probka_krotka.shape}")
    print(f"Rozmiar wyjściowego embeddingu: {emb1.shape}\n")

    print("--- TEST PRÓBKI 2 (Długie słowo: 32x256) ---")
    # Tworzymy sztuczny obraz (Batch=1, Kanał=1, Wysokość=32, Szerokość=256)
    probka_dluga = torch.randn(1, 1, 32, 256)
    emb2 = model.forward_once(probka_dluga)
    print(f"Rozmiar wejściowy: {probka_dluga.shape}")
    print(f"Rozmiar wyjściowego embeddingu: {emb2.shape}\n")
    
    # Porównanie (Sieć Syjamska)
    # W fazie uczenia obliczałbyś teraz odległość między emb1 a emb2
    dist = torch.pairwise_distance(emb1, emb2)
    print(f"Odległość euklidesowa między wektorami cech: {dist.item():.4f}")

    pretrained_encoder = pretrain_model()


if __name__ == "__main__":
    main()
