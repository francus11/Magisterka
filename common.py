from sklearn.model_selection import train_test_split


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