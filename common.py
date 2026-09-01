from sklearn.model_selection import train_test_split
import numpy as np


GLOBAL_SEED = 42

def build_word_dataframe(df, pics_path):
	df_selected = df[["word_id", "word_text", "user_class"]].copy()
	df_selected = df_selected.explode(["word_id", "word_text"], ignore_index=True)
	df_selected["word_path"] = pics_path + "/" + df_selected["word_id"].astype(str) + ".png"
	return df_selected

def get_subsets(df, test_size=0.2, val_size=0.20, random_state=GLOBAL_SEED):
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

def get_balanced_disjoint_subsets(df, test_size=0.2, val_size=0.2, random_state=GLOBAL_SEED):
    """
    Dzieli zbiór tak, aby klasy były w 100% rozłączne,
    a łączna LICZBA PRÓBEK (wierszy) w train/val/test odpowiadała zadanym proporcjom.
    """
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1.0")

    train_size = 1.0 - test_size - val_size
    total_samples = len(df)
    
    # Docelowa liczba próbek dla każdego podzbioru
    target_counts = {
        'train': int(total_samples * train_size),
        'val': int(total_samples * val_size),
        'test': int(total_samples * test_size)
    }

    # 1. Zlicz próbki dla każdego autora i przetasuj (dla powtarzalności)
    class_counts = df['user_class'].value_counts()
    
    # Tasowanie autorów o tej samej liczbie próbek (reproducibility)
    rng = np.random.default_rng(random_state)
    shuffled_classes = rng.permutation(class_counts.index)
    sorted_classes = sorted(shuffled_classes, key=lambda c: class_counts[c], reverse=True)

    assigned_classes = {'train': [], 'val': [], 'test': []}
    current_counts = {'train': 0, 'val': 0, 'test': 0}

    # 2. Zachłanny przydział autorów do podzbioru z największym deficytem próbek
    for cls in sorted_classes:
        count = class_counts[cls]
        
        # Wybierz podzbiór, któremu najbardziej brakuje próbek do celu
        best_split = max(
            ['train', 'val', 'test'],
            key=lambda s: (target_counts[s] - current_counts[s])
        )
        
        assigned_classes[best_split].append(cls)
        current_counts[best_split] += count

    # 3. Przefiltruj DataFrame
    df_train = df[df['user_class'].isin(assigned_classes['train'])].copy().reset_index(drop=True)
    df_val = df[df['user_class'].isin(assigned_classes['val'])].copy().reset_index(drop=True)
    df_test = df[df['user_class'].isin(assigned_classes['test'])].copy().reset_index(drop=True)

    # Raport z podziału
    print(f"--- Podział próbek ---")
    print(f"Train: {len(df_train)} próbek ({len(df_train)/total_samples*100:.1f}%) | {len(assigned_classes['train'])} klas")
    print(f"Val:   {len(df_val)} próbek ({len(df_val)/total_samples*100:.1f}%) | {len(assigned_classes['val'])} klas")
    print(f"Test:  {len(df_test)} próbek ({len(df_test)/total_samples*100:.1f}%) | {len(assigned_classes['test'])} klas")

    return df_train, df_val, df_test