import pandas as pd
import numpy as np
import time
from datetime import datetime


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2

import common

from sklearn.model_selection import train_test_split

from crnn import CRNNEncoder

class InfiniteSiameseDataset(Dataset):
    def __init__(self, df, samples_per_epoch=20000):
        self.df = df
        self.samples_per_epoch = samples_per_epoch
        self.author_groups = df.groupby('user_class').indices
        self.authors = list(self.author_groups.keys())

    def __len__(self):
        # Narzucamy DataLoaderowi, ile kroków ma liczyć jedna epoka
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Logika dynamicznego losowania pary (zawsze świeże losowanie)
        # ...
        return img1, img2, label
    
class SiameseHandwritingDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.author_groups = df.groupby('user_class').indices
        self.authors = list(self.author_groups.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Losujemy: 0 -> ta sama osoba, 1 -> różna osoba
        same_author = np.random.rand() > 0.5
        
        row1 = self.df.iloc[idx]
        author1 = row1['user_class']
        
        if same_author and len(self.author_groups[author1]) > 1:
            idx2 = np.random.choice(self.author_groups[author1])
            label = 0.0
        else:
            diff_author = np.random.choice([a for a in self.authors if a != author1])
            idx2 = np.random.choice(self.author_groups[diff_author])
            label = 1.0
            
        row2 = self.df.iloc[idx2]
        
        # Wczytanie obrazów (funkcja pomocnicza load_img zwraca tensor [1, H, W])
        img1 = self._load_img(row1['word_path'])
        img2 = self._load_img(row2['word_path'])
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)

    def _load_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)

class SiameseNetwork(nn.Module):
    def __init__(self, pretrained_model):
        super(SiameseNetwork, self).__init__()
        self.encoder = pretrained_model

    def forward_one(self, x):
        emb = self.encoder(x)
        # Normalizacja L2 rzutuje wektory na hipersferę jednostkową
        # Ułatwia to porównywanie miarami euklidesowymi i kosinusowymi
        return F.normalize(emb, p=2, dim=1)

    def forward(self, img1, img2):
        emb1 = self.forward_one(img1)
        emb2 = self.forward_one(img2)
        return emb1, emb2
    
class CachedSiameseDataset(Dataset):
    """
    Trzyma obrazy w RAM i dynamicznie generuje zbalansowane pary:
    - 50% par pozytywnych (ten sam autor, label=0)
    - 50% par negatywnych (różni autorzy, label=1)
    """
    def __init__(self, df, samples_per_epoch=10000):
        self.samples_per_epoch = samples_per_epoch
        
        print("Wczytywanie próbek dla sieci syjamskiej do RAM...")
        self.images = []
        self.labels = df["user_class"].tolist()
        
        for path in df["word_path"].tolist():
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Nie można wczytać pliku: {path}")
            tensor_img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
            self.images.append(tensor_img)
            
        # Tworzymy mapę {autor_id: [lista indeksów próbek tego autora]}
        self.author_to_indices = {}
        for idx, author_id in enumerate(self.labels):
            self.author_to_indices.setdefault(author_id, []).append(idx)
            
        self.authors = list(self.author_to_indices.keys())
        print(f"Wczytano {len(self.images)} próbek od {len(self.authors)} unikalnych autorów.")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Losujemy typ pary: 0 -> ten sam autor, 1 -> różni autorzy
        should_be_same = np.random.rand() > 0.5
        
        # Wybieramy pierwszego autora i pierwszą próbkę
        author1 = np.random.choice(self.authors)
        idx1 = np.random.choice(self.author_to_indices[author1])
        
        if should_be_same and len(self.author_to_indices[author1]) > 1:
            # Ta sama osoba: wybieramy INNE słowo tej samej osoby
            possible_indices = [i for i in self.author_to_indices[author1] if i != idx1]
            idx2 = np.random.choice(possible_indices) if possible_indices else idx1
            label = 0.0
        else:
            # Różne osoby: losujemy innego autora
            author2 = np.random.choice([a for a in self.authors if a != author1])
            idx2 = np.random.choice(self.author_to_indices[author2])
            label = 1.0

        return self.images[idx1], self.images[idx2], torch.tensor(label, dtype=torch.float32)
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        # Odległość euklidesowa
        euclidean_distance = F.pairwise_distance(emb1, emb2)
        
        # Strata dla par pozytywnych (ta sama osoba: label = 0)
        loss_pos = (1 - label) * torch.pow(euclidean_distance, 2)
        # Strata dla par negatywnych (różne osoby: label = 1)
        loss_neg = label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        loss = torch.mean(loss_pos + loss_neg)
        return loss

def collate_fn_siamese_pad(batch):
    """Dopełnia paddingiem obie paczki obrazów (img1 i img2) niezależnie."""
    imgs1, imgs2, labels = zip(*batch)
    
    # 1. Padding dla pierwszej grupy obrazów
    max_w1 = max(img.shape[-1] for img in imgs1)
    padded_imgs1 = [nn.functional.pad(img, (0, max_w1 - img.shape[-1], 0, 0), value=0) for img in imgs1]
    
    # 2. Padding dla drugiej grupy obrazów
    max_w2 = max(img.shape[-1] for img in imgs2)
    padded_imgs2 = [nn.functional.pad(img, (0, max_w2 - img.shape[-1], 0, 0), value=0) for img in imgs2]
    
    return torch.stack(padded_imgs1), torch.stack(padded_imgs2), torch.tensor(labels)

def worker_init_fn(worker_id):
    """Gwarantuje unikalne ziarno losowe dla każdego wątku i każdej epoki."""
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    
def main():
    EPOCHS = 10
    SAMPLES_PER_EPOCH = 20000
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    MARGIN = 1.0
    THRESHOLD = 0.5
    
    folder_prefix = "siamese_training_"
    current_datetime = datetime.now()
    
    folder_name = f"{folder_prefix}_{current_datetime.strftime('%Y%m%d_%H%M%S')}"
    
    df_words = pd.read_parquet("df_words_preprocessed.parquet")
    
    classes = sorted(df_words["user_class"].unique())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    df_words["user_class"] = df_words["user_class"].map(class_to_idx)
    
    df_train, df_val, df_test = common.get_subsets(df_words, test_size=0.2, val_size=0.20)
    
    
    pretrained_model = CRNNEncoder()
          
    pretrained_path = "crnn_encoder_pretrained_70percent.pth"
    # Wczytanie wag z pre-trainingu
    if pretrained_path:
        pretrained_model.load_state_dict(torch.load(pretrained_path))
        print(f"Wczytano wagi enkodera z {pretrained_path}")
    
    model = SiameseNetwork(pretrained_model)
    criterion = ContrastiveLoss(margin=MARGIN)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    train_dataset = CachedSiameseDataset(df_train, samples_per_epoch=SAMPLES_PER_EPOCH)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,                      # shuffle wymusza nowe seedy PyTorcha w każdej epoce
        worker_init_fn=worker_init_fn,     # Rozwiązuje problem duplikacji
        collate_fn=collate_fn_siamese_pad,
    )
    
    print("--- ROZPOCZĘCIE ETAPU UCZENIA SIAMESE ---")
    model.train()
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (img1, img2, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * img1.size(0)
            total += labels.size(0)
            # --- OBLICZANIE ACCURACY ---
            with torch.no_grad():
                # Dystans euklidesowy między wektorami w batchu: [Batch_Size]
                distances = F.pairwise_distance(emb1, emb2)
                
                # Predykcja: 1.0 jeśli dystans >= THRESHOLD (różni), inaczej 0.0 (ci sami)
                predictions = (distances >= THRESHOLD).float()
                
                # Zliczanie poprawnych trafień
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            
        epoch_loss = running_loss / SAMPLES_PER_EPOCH
        current_lr = scheduler.get_last_lr()[0]
        epoch_acc = (correct / total) * 100
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoka [{epoch+1}/{EPOCHS}] | Straty (Loss): {epoch_loss:.4f} | LR: {current_lr:.6f} | Dokładność (Acc): {epoch_acc:.2f}% | Czas: {epoch_time:.2f}s")

    torch.save(model.encoder.state_dict(), "crnn_siamese.pth")
    print("Zapisano wagi wyuczonego enkodera do pliku: 'crnn_siamese.pth'")

    print("--- ZAKOŃCZONO UCZENIE SIAMESE ---")
    return model.encoder
    
if __name__ == "__main__":
    main()