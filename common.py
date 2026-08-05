GLOBAL_SEED = 42

def build_word_dataframe(df, pics_path):
	df_selected = df[["word_id", "word_text", "user_class"]].copy()
	df_selected = df_selected.explode(["word_id", "word_text"], ignore_index=True)
	df_selected["word_path"] = pics_path + "/" + df_selected["word_id"].astype(str) + ".png"
	return df_selected