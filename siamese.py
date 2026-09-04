import pandas as pd
import numpy as np
import time
import random
import json
from pathlib import Path
from datetime import datetime


from pre_trained_models import ResNetEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2

import common
from crnn import CRNNEncoder

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


def save_checkpoint(path, epoch, max_epochs, model, optimizer, scheduler, metrics):
    """Zapisuje kompletny stan treningu po ukończonej epoce."""
    checkpoint = {
        "epoch": epoch,
        "max_epochs": max_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    torch.save(checkpoint, path)


def save_session_settings(path, settings):
    """Zapisuje ustawienia użyte podczas całej sesji treningowej."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2, ensure_ascii=False)


def load_checkpoint(path, model, optimizer, scheduler):
    """Wczytuje ostatni kompletny stan treningu."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    torch.set_rng_state(checkpoint["torch_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])

    return checkpoint["epoch"], checkpoint.get("max_epochs"), checkpoint.get("metrics", [])
    
def train_siamese(
    resume_dir = None,
    max_epochs = 10,
    samples_per_epoch = 20000,
    batch_size = 8,
    dropout_p = 0.0,
    learning_rate = 1e-3,
    margin = 1.0,
    threshold = 0.5,
    weight_decay = 0.0,
    pretrained_encoder = None
    ):
    # =============================================
    # Przygotowanie zmiennych dla session_settings
    # =============================================
    # Te zmienne będą używane do inicjalizacji session_settings
    initial_max_epochs = max_epochs
    initial_samples_per_epoch = samples_per_epoch
    initial_batch_size = batch_size
    initial_learning_rate = learning_rate
    initial_margin = margin
    initial_threshold = threshold
    
    # =============================================
    # Load and preprocess the dataset
    # =============================================
    #region
    
    df_words = pd.read_parquet("df_words_preprocessed.parquet")
    
    classes = sorted(df_words["user_class"].unique())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    df_words["user_class"] = df_words["user_class"].map(class_to_idx)
    
    df_train, df_val, df_test = common.get_balanced_disjoint_subsets(df_words, test_size=0.2, val_size=0.20)
    
    train_dataset = CachedSiameseDataset(df_train, samples_per_epoch=samples_per_epoch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, collate_fn=collate_fn_siamese_pad)
    val_dataset = CachedSiameseDataset(df_val, samples_per_epoch=int(samples_per_epoch/5))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, collate_fn=collate_fn_siamese_pad)
    #endregion
    
    # =============================================
    # Initialize the Siamese Network and training components
    # =============================================
    #region
    
    if pretrained_encoder is not None:
        pretrained_model = pretrained_encoder
    else:
        pretrained_model = CRNNEncoder()
            
        pretrained_path = "crnn_encoder_pretrained_70percent.pth"
        # Wczytanie wag z pre-trainingu
        if pretrained_path:
            pretrained_model.load_state_dict(torch.load(pretrained_path))
            print(f"Wczytano wagi enkodera z {pretrained_path}")
    
    model = SiameseNetwork(pretrained_model)
    
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    #endregion
    
    # =============================================
    # Prepare output directories and handle resuming from checkpoint
    # =============================================
    #region
    
    if resume_dir is None:
        # training_siamese_[model]_[pretrained_encoder]_[optimizer]_[date]
        folder_name = f"training_siamese_{type(model).__name__}_{type(pretrained_model).__name__}_{type(optimizer).__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = Path(folder_name)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(resume_dir)
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Nie znaleziono folderu: {output_dir}")

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
    latest_checkpoint = output_dir / "latest.pt"
    metrics_path = output_dir / "metrics.csv"
    settings_path = output_dir / "session_settings.json"

    if resume_dir is None:
        session_settings = {
            "max_epochs": max_epochs,
            "samples_per_epoch": samples_per_epoch,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "margin": margin,
            "threshold": threshold,
            "pretrained_path": "crnn_encoder_pretrained_70percent.pth",
            "weight_decay": 1e-4,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "enable_parameter_override": False,
            "parameter_overrides": {},
        }
    else:
        if settings_path.exists():
            with settings_path.open("r", encoding="utf-8") as file:
                session_settings = json.load(file)
            max_epochs = session_settings["max_epochs"]
            threshold = session_settings["threshold"]

    if resume_dir is None:
        session_settings.update({
            "model": type(model).__name__,
            "encoder": type(pretrained_model).__name__,
            "optimizer": type(optimizer).__name__,
            "scheduler": type(scheduler).__name__,
        })
        save_session_settings(settings_path, session_settings)

    if resume_dir is not None:
        if not latest_checkpoint.exists():
            raise FileNotFoundError(f"Nie znaleziono checkpointu: {latest_checkpoint}")

        # Sprawdzenie czy należy nadpisać parametry
        if settings_path.exists():
            with settings_path.open("r", encoding="utf-8") as file:
                session_settings = json.load(file)
            
            if session_settings.get("enable_parameter_override", False):
                overrides = session_settings.get("parameter_overrides", {})
                if overrides:
                    print("\n=== NADPISYWANIE PARAMETRÓW ===")
                    batch_size_changed = False
                    samples_per_epoch_changed = False
                    
                    if "max_epochs" in overrides:
                        old_val = max_epochs
                        max_epochs = overrides["max_epochs"]
                        print(f"max_epochs: {old_val} -> {max_epochs}")
                    if "samples_per_epoch" in overrides:
                        old_val = samples_per_epoch
                        samples_per_epoch = overrides["samples_per_epoch"]
                        samples_per_epoch_changed = True
                        print(f"samples_per_epoch: {old_val} -> {samples_per_epoch}")
                    if "batch_size" in overrides:
                        old_val = batch_size
                        batch_size = overrides["batch_size"]
                        batch_size_changed = True
                        print(f"batch_size: {old_val} -> {batch_size}")
                    if "learning_rate" in overrides:
                        old_val = learning_rate
                        learning_rate = overrides["learning_rate"]
                        print(f"learning_rate: {old_val} -> {learning_rate}")
                        # Zaktualizuj optimizer
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate
                    if "margin" in overrides:
                        old_val = margin
                        margin = overrides["margin"]
                        print(f"margin: {old_val} -> {margin}")
                    if "threshold" in overrides:
                        old_val = threshold
                        threshold = overrides["threshold"]
                        print(f"threshold: {old_val} -> {threshold}")
                    if "weight_decay" in overrides:
                        old_val = weight_decay
                        weight_decay = overrides["weight_decay"]
                        print(f"weight_decay: {old_val} -> {weight_decay}")
                        # Zaktualizuj optimizer
                        for param_group in optimizer.param_groups:
                            param_group['weight_decay'] = weight_decay
                    print("==========================\n")
                    
                    # Jeśli samples_per_epoch lub batch_size się zmieniły, stwórz nowe datasety i dataloadery
                    if samples_per_epoch_changed or batch_size_changed:
                        print("Rekonstrukcja datasetów i dataloaderów...")
                        train_dataset = CachedSiameseDataset(df_train, samples_per_epoch=samples_per_epoch)
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, collate_fn=collate_fn_siamese_pad)
                        val_dataset = CachedSiameseDataset(df_val, samples_per_epoch=int(samples_per_epoch/5))
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, collate_fn=collate_fn_siamese_pad)
                        print("Datasety i dataloadery zostały zaktualizowane.\n")

        start_epoch, checkpoint_max_epochs, metrics_history = load_checkpoint(
            latest_checkpoint,
            model,
            optimizer,
            scheduler,
        )
        if not settings_path.exists() and checkpoint_max_epochs is not None:
            max_epochs = checkpoint_max_epochs
        if max_epochs != scheduler.T_max:
            scheduler.T_max = max_epochs
        print(f"Wznowiono trening od epoki {start_epoch + 1}.")
        
    #endregion

    # =============================================
    # Learning loop
    # =============================================
    #region
    
    if resume_dir is None:
        start_epoch = 0
        metrics_history = []
    
    print("--- ROZPOCZĘCIE ETAPU UCZENIA SIAMESE ---")
    model.train()
    
    for epoch in range(start_epoch, max_epochs):
        # ===== training ===== 
        
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_true_positive = 0
        train_true_negative = 0
        train_false_positive = 0
        train_false_negative = 0
        
        model.train()
        train_epoch_start_time = time.time()
        
        for batch_idx, (img1, img2, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, labels)
            
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item() * img1.size(0)

            # --- OBLICZANIE ACCURACY ---
            with torch.no_grad():
                # Dystans euklidesowy między wektorami w batchu: [Batch_Size]
                distances = F.pairwise_distance(emb1, emb2)
                
                # Predykcja: 1.0 jeśli dystans >= THRESHOLD (różni), inaczej 0.0 (ci sami)
                predictions = (distances >= threshold).float()
                
                # Zliczanie poprawnych trafień
                train_correct += (predictions == labels).sum().item()
                
                train_true_positive += ((predictions == 1.0) & (labels == 1.0)).sum().item()
                train_true_negative += ((predictions == 0.0) & (labels == 0.0)).sum().item()
                train_false_positive += ((predictions == 1.0) & (labels == 0.0)).sum().item()
                train_false_negative += ((predictions == 0.0) & (labels == 1.0)).sum().item()
                
                train_total += labels.size(0)
        
        train_epoch_time = time.time() - train_epoch_start_time
        
        train_epoch_loss = train_running_loss / samples_per_epoch
        train_epoch_acc = (train_correct / train_total) * 100
        
        # ===== validate ===== 
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_true_positive = 0
        val_true_negative = 0
        val_false_positive = 0
        val_false_negative = 0
        
        model.eval()
        val_epoch_start_time = time.time()
        for batch_idx, (img1, img2, labels) in enumerate(val_loader):
            with torch.no_grad():
                emb1, emb2 = model(img1, img2)
                loss = criterion(emb1, emb2, labels)
                val_running_loss += loss.item() * img1.size(0)

                distances = F.pairwise_distance(emb1, emb2)
                predictions = (distances >= threshold).float()
                
                val_correct += (predictions == labels).sum().item()
                val_true_positive += ((predictions == 1.0) & (labels == 1.0)).sum().item()
                val_true_negative += ((predictions == 0.0) & (labels == 0.0)).sum().item()
                val_false_positive += ((predictions == 1.0) & (labels == 0.0)).sum().item()
                val_false_negative += ((predictions == 0.0) & (labels == 1.0)).sum().item()
                
                val_total += labels.size(0)
            
        val_epoch_loss = val_running_loss / samples_per_epoch
        val_epoch_acc = (val_correct / val_total) * 100
        current_lr = optimizer.param_groups[0]["lr"]
        
        # ===== metrics ===== 
        
        val_epoch_time = time.time() - val_epoch_start_time
        epoch_metrics = {
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "dropout_p": session_settings.get("dropout_p", 0.0),
            "weight_decay": weight_decay,
            "train_loss": train_epoch_loss,
            "train_accuracy": train_epoch_acc,
            "train_time_seconds": train_epoch_time,
            "train_true_positive": train_true_positive,
            "train_true_negative": train_true_negative,
            "train_false_positive": train_false_positive,
            "train_false_negative": train_false_negative,
            "val_loss": val_epoch_loss,
            "val_accuracy": val_epoch_acc,
            "val_time_seconds": val_epoch_time,
            "val_true_positive": val_true_positive,
            "val_true_negative": val_true_negative,
            "val_false_positive": val_false_positive,
            "val_false_negative": val_false_negative,
        }
        metrics_history.append(epoch_metrics)

        # Scheduler musi zostać przesunięty po zakończeniu epoki.
        scheduler.step()

        # Metryki są zapisywane po każdej ukończonej epoce.
        pd.DataFrame(metrics_history).to_csv(metrics_path, index=False)

        # Osobny plik archiwalny dla każdej epoki.
        save_checkpoint(
            checkpoints_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt",
            epoch + 1,
            max_epochs,
            model,
            optimizer,
            scheduler,
            metrics_history,
        )

        # Ten plik służy do wznowienia treningu.
        save_checkpoint(
            latest_checkpoint,
            epoch + 1,
            max_epochs,
            model,
            optimizer,
            scheduler,
            metrics_history,
        )

        print(f"Epoka [{epoch+1}/{max_epochs}] | Straty (Loss): {train_epoch_loss:.4f} | LR: {current_lr:.6f} | Dokładność (Acc): {train_epoch_acc:.2f}% | Czas: {train_epoch_time:.2f}s")
        
    #endregion
        
    # =============================================
    # Return final encoder weights and save them
    # =============================================

    torch.save(model.encoder.state_dict(), output_dir / "crnn_siamese.pth")
    print(f"Zapisano wagi wyuczonego enkodera w folderze: '{output_dir}'")

    print("--- ZAKOŃCZONO UCZENIE SIAMESE ---")
    return model.encoder

def test_siamese(model, df_test, threshold=0.5):
    model.eval()
    test_dataset = CachedSiameseDataset(df_test, samples_per_epoch=len(df_test))
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn_siamese_pad,
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for img1, img2, labels in test_loader:
            emb1, emb2 = model(img1, img2)
            distances = F.pairwise_distance(emb1, emb2)
            predictions = (distances >= threshold).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = (correct / total) * 100
    print(f"Dokładność na zbiorze testowym: {accuracy:.2f}%")
    return accuracy
    
if __name__ == "__main__":
    train_siamese()
    # test_model = SiameseNetwork(ResNetEncoder(embedding_dim=128, pretrained=True))
    # test_model.encoder.load_state_dict(torch.load("crnn_siamese.pth"))
    # df_words = pd.read_parquet("df_words_another_preprocessed.parquet").head(20000)
    # test_siamese(test_model, df_words, threshold=0.5)