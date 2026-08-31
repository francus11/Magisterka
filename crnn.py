import pandas as pd
import numpy as np
import time
import random
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import cv2

import common # mój moduł z powszechnymi funkcjami i stałymi
from pre_trained_models import ResNetEncoder

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

class PretrainResnetClassifier(nn.Module):
    """Pełny model z nagłówkiem klasyfikacyjnym używany TYLKO do pre-trainingu."""
    def __init__(self, num_classes, embedding_dim=128):
        super(PretrainResnetClassifier, self).__init__()
        self.encoder = ResNetEncoder(embedding_dim=embedding_dim)
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


def save_checkpoint(path, epoch, max_epochs, model, optimizer, metrics, generator):
    """Zapisuje kompletny stan pre-trainingu po ukończonej epoce."""
    checkpoint = {
        "epoch": epoch,
        "max_epochs": max_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "dataloader_generator_state": generator.get_state(),
    }
    torch.save(checkpoint, path)


def save_session_settings(path, settings):
    """Zapisuje ustawienia użyte podczas sesji treningowej."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2, ensure_ascii=False)


def load_checkpoint(path, model, optimizer, generator):
    """Wczytuje ostatni kompletny stan pre-trainingu."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    torch.set_rng_state(checkpoint["torch_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    generator.set_state(checkpoint["dataloader_generator_state"])

    return checkpoint["epoch"], checkpoint.get("max_epochs"), checkpoint.get("metrics", [])

def train_pretrain(
    embedding_dim = 128,   # Rozmiar docelowego wektora cech
    batch_size = 8,
    max_epochs = 10,
    learning_rate = 1e-3,
    resume_dir = None,
    model = None
):
    # =============================================
    # Load and preprocess the dataset
    # =============================================
    #region
    
    df_words = pd.read_parquet("df_words_preprocessed.parquet")
    
    classes = sorted(df_words["user_class"].unique())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    df_words["user_class"] = df_words["user_class"].map(class_to_idx)
    
    df_train, df_val, df_test = common.get_subsets(df_words, test_size=0.2, val_size=0.20)

    train_dataset = HandwritingDataset(df_train)
    val_dataset = HandwritingDataset(df_val)
    test_dataset = HandwritingDataset(df_test)
    
    generator = torch.Generator().manual_seed(common.GLOBAL_SEED)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, collate_fn=collate_fn_pad)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=generator, collate_fn=collate_fn_pad)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=generator, collate_fn=collate_fn_pad)

    #endregion
    
    # =============================================
    # Initialize classifier and training components
    # =============================================
    #region
    
    # Inicjalizacja modelu, funkcji straty i optymalizatora
    if model is None:
        model = PretrainResnetClassifier(num_classes=len(df_train["user_class"].unique()), embedding_dim=embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    #endregion
    
        
    # =============================================
    # Prepare output directories and handle resuming from checkpoint
    # =============================================
    #region
    
    if resume_dir is None:
        # training_crnn_[model]_[encoder]_[optimizer]_[date]
        output_dir = Path(f"training_crnn_{type(model).__name__}_{type(model.encoder).__name__}_{type(optimizer).__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(resume_dir)
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Nie znaleziono folderu: {output_dir}")

    latest_checkpoint = output_dir / "latest.pt"
    metrics_path = output_dir / "metrics.csv"
    settings_path = output_dir / "session_settings.json"

    if resume_dir is not None and settings_path.exists():
        with settings_path.open("r", encoding="utf-8") as file:
            session_settings = json.load(file)
        max_epochs = session_settings["max_epochs"]

    if resume_dir is None:
        session_settings = {
            "max_epochs": max_epochs,
            "embedding_dim": embedding_dim,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dataset": type(train_dataset).__name__,
            "model": type(model).__name__,
            "encoder": type(model.encoder).__name__,
            "criterion": type(criterion).__name__,
            "optimizer": type(optimizer).__name__,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_session_settings(settings_path, session_settings)

    if resume_dir is not None:
        if not latest_checkpoint.exists():
            raise FileNotFoundError(f"Nie znaleziono checkpointu: {latest_checkpoint}")

        start_epoch, checkpoint_max_epochs, metrics_history = load_checkpoint(
            latest_checkpoint,
            model,
            optimizer,
            generator,
        )
        if not settings_path.exists() and checkpoint_max_epochs is not None:
            max_epochs = checkpoint_max_epochs
        print(f"Wznowiono pre-training od epoki {start_epoch + 1}.")
        
    #endregion
        
    # =============================================
    # Learning loop
    # =============================================
    #region

    start_epoch = 0
    metrics_history = []
    
    print("--- ROZPOCZĘCIE ETAPU PRE-TRAININGU ---")
    model.train()
    
    for epoch in range(start_epoch, max_epochs):
        
        running_loss = 0.0
        correct = 0
        total = 0

        # ===== train =====
        model.train()
        train_epoch_start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(train_dataloader):
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

        train_epoch_time = time.time() - train_epoch_start_time
        
        train_epoch_loss = running_loss / total
        train_epoch_acc = (correct / total) * 100
        
        # ===== validate =====
        model.eval()
        val_epoch_start_time = time.time()
        
        with torch.no_grad():
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0

            for images, labels in val_dataloader:
                logits, _ = model(images)
                loss = criterion(logits, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

            val_epoch_loss = val_running_loss / val_total
            val_epoch_acc = (val_correct / val_total) * 100
        
        val_epoch_time = time.time() - val_epoch_start_time
        
        # ===== metrics =====
        epoch_metrics = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_epoch_loss,
            "train_accuracy": train_epoch_acc,
            "train_time_seconds": train_epoch_time,
            "val_loss": val_epoch_loss,
            "val_accuracy": val_epoch_acc,
            "val_time_seconds": val_epoch_time,
        }
        metrics_history.append(epoch_metrics)

        pd.DataFrame(metrics_history).to_csv(metrics_path, index=False)
        save_checkpoint(
            output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt",
            epoch + 1,
            max_epochs,
            model,
            optimizer,
            metrics_history,
            generator,
        )
        save_checkpoint(
            latest_checkpoint,
            epoch + 1,
            max_epochs,
            model,
            optimizer,
            metrics_history,
            generator,
        )

        print(f"Epoka [{epoch+1}/{max_epochs}] TRAIN | Straty (Loss): {train_epoch_loss:.4f} | Dokładność (Acc): {train_epoch_acc:.2f}% | Czas: {train_epoch_time:.2f}s")
        print(f"Epoka [{epoch+1}/{max_epochs}] VALIDATION | Straty (Loss): {val_epoch_loss:.4f} | Dokładność (Acc): {val_epoch_acc:.2f}% | Czas: {val_epoch_time:.2f}s")

    #endregion
        
    # =============================================
    # Return final encoder weights and save them
    # =============================================

    print("\n--- ZAKOŃCZONO PRE-TRAINING ---")

    torch.save(model.encoder.state_dict(), output_dir / f"{model.__class__.__name__}_encoder_pretrained.pth")
    print(f"Zapisano wagi wyuczonego enkodera w folderze: '{output_dir}'")

    return model.encoder

if __name__ == "__main__":
    train_pretrain()
