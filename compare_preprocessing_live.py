import base64
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import pandas as pd

from preprocessing import build_word_dataframe, preprocess_handwriting_image


DATASET_WORDS_DIR = "dataset_words/words"
PREVIEW_MAX_SIDE = 360


def resize_to_fit(image, max_side=PREVIEW_MAX_SIDE):
	if image is None:
		return None

	height, width = image.shape[:2]
	longest_side = max(height, width)
	if longest_side <= max_side:
		return image

	scale = max_side / float(longest_side)
	new_width = max(1, int(round(width * scale)))
	new_height = max(1, int(round(height * scale)))
	return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def image_to_photoimage(image):
	if image is None:
		return None

	array = image
	if array.dtype != np.uint8:
		array = (array * 255.0).clip(0, 255).astype(np.uint8)

	encoded = cv2.imencode(".png", array)[1].tobytes()
	return tk.PhotoImage(data=base64.b64encode(encoded).decode("ascii"))


class PreprocessingComparer:
	def __init__(self, root):
		self.root = root
		self.root.title("Porownanie oryginalu i preprocessingu")
		self.root.geometry("1180x760")

		self.df_selected = self._load_dataset()
		self.current_image = None
		self.original_photo = None
		self.processed_photo = None
		self.refresh_job = None

		self.current_path = tk.StringVar()
		self.current_word = tk.StringVar(value="")
		self.status_text = tk.StringVar(value="Wybierz obraz, aby rozpocząć.")
		self.target_width = tk.IntVar(value=128)
		self.target_height = tk.IntVar(value=128)
		self.blur_kernel_size = tk.IntVar(value=3)
		self.threshold_block_size = tk.IntVar(value=31)
		self.threshold_c = tk.IntVar(value=11)
		self.invert_threshold = tk.BooleanVar(value=True)
		self.adaptive_method = tk.StringVar(value="gaussian")

		self._build_ui()
		self._bind_changes()

		if not self.df_selected.empty:
			self._load_row(0)
		else:
			self._refresh_views()

	def _load_dataset(self):
		if not os.path.exists("df.parquet"):
			messagebox.showerror("Blad", "Brakuje pliku df.parquet w katalogu projektu.")
			return pd.DataFrame()

		df = pd.read_parquet("df.parquet")
		return build_word_dataframe(df)

	def _build_ui(self):
		container = ttk.Frame(self.root, padding=12)
		container.pack(fill=tk.BOTH, expand=True)

		top_row = ttk.Frame(container)
		top_row.pack(fill=tk.X, pady=(0, 10))

		ttk.Label(top_row, text="Plik obrazu").pack(side=tk.LEFT)
		self.path_entry = ttk.Entry(top_row, textvariable=self.current_path, width=76)
		self.path_entry.pack(side=tk.LEFT, padx=(8, 8), fill=tk.X, expand=True)
		ttk.Button(top_row, text="Otworz plik...", command=self._browse_file).pack(side=tk.LEFT)

		if not self.df_selected.empty:
			sample_values = [f"{row.word_id} | {row.word_text}" for row in self.df_selected.head(500).itertuples()]
			self.sample_choice = tk.StringVar(value=sample_values[0] if sample_values else "")
			self.sample_combo = ttk.Combobox(top_row, textvariable=self.sample_choice, values=sample_values, width=28, state="readonly")
			self.sample_combo.pack(side=tk.LEFT, padx=(8, 0))
			self.sample_combo.bind("<<ComboboxSelected>>", self._on_sample_selected)

		main = ttk.Frame(container)
		main.pack(fill=tk.BOTH, expand=True)

		controls = ttk.LabelFrame(main, text="Parametry", padding=12)
		controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

		self._add_scale(controls, "Szerokosc docelowa", self.target_width, 32, 512)
		self._add_scale(controls, "Wysokosc docelowa", self.target_height, 32, 512)
		self._add_scale(controls, "Blur kernel", self.blur_kernel_size, 1, 15)
		self._add_scale(controls, "Block size", self.threshold_block_size, 3, 51)
		self._add_scale(controls, "Threshold C", self.threshold_c, -30, 30)

		method_frame = ttk.Frame(controls)
		method_frame.pack(fill=tk.X, pady=(10, 6))
		ttk.Label(method_frame, text="Adaptive method").pack(anchor=tk.W)
		method_choice = ttk.Combobox(method_frame, textvariable=self.adaptive_method, values=["gaussian", "mean"], state="readonly")
		method_choice.pack(fill=tk.X, pady=(2, 0))

		invert_frame = ttk.Frame(controls)
		invert_frame.pack(fill=tk.X, pady=(6, 10))
		ttk.Checkbutton(invert_frame, text="Invert threshold", variable=self.invert_threshold).pack(anchor=tk.W)

		actions = ttk.Frame(controls)
		actions.pack(fill=tk.X, pady=(10, 0))
		ttk.Button(actions, text="Odśwież", command=self._refresh_views).pack(fill=tk.X)
		ttk.Button(actions, text="Nastepna probka", command=self._load_next_sample).pack(fill=tk.X, pady=(6, 0))

		preview_area = ttk.Frame(main)
		preview_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

		self.original_card = self._create_preview_card(preview_area, "Oryginal")
		self.original_card.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
		self.processed_card = self._create_preview_card(preview_area, "Po preprocessingu")
		self.processed_card.pack(fill=tk.BOTH, expand=True)

		status_bar = ttk.Label(container, textvariable=self.status_text, anchor=tk.W)
		status_bar.pack(fill=tk.X, pady=(10, 0))

	def _create_preview_card(self, parent, title):
		card = ttk.LabelFrame(parent, text=title, padding=12)
		label = ttk.Label(card)
		label.pack(fill=tk.BOTH, expand=True)
		meta = ttk.Label(card, text="", anchor=tk.W)
		meta.pack(fill=tk.X, pady=(8, 0))
		card.image_label = label
		card.meta_label = meta
		return card

	def _add_scale(self, parent, title, variable, from_value, to_value):
		frame = ttk.Frame(parent)
		frame.pack(fill=tk.X, pady=(0, 6))
		ttk.Label(frame, text=title).pack(anchor=tk.W)
		scale = tk.Scale(
			frame,
			from_=from_value,
			to=to_value,
			orient=tk.HORIZONTAL,
			resolution=1,
			showvalue=True,
			variable=variable,
			length=260,
		)
		scale.pack(fill=tk.X)

	def _bind_changes(self):
		for variable in [
			self.target_width,
			self.target_height,
			self.blur_kernel_size,
			self.threshold_block_size,
			self.threshold_c,
			self.invert_threshold,
			self.adaptive_method,
		]:
			variable.trace_add("write", self._schedule_refresh)

	def _schedule_refresh(self, *_args):
		if self.refresh_job is not None:
			self.root.after_cancel(self.refresh_job)
		self.refresh_job = self.root.after(120, self._refresh_views)

	def _load_row(self, index):
		row = self.df_selected.iloc[index]
		self.current_path.set(row.word_path)
		self.current_word.set(f"{row.word_id} | {row.word_text}")
		if hasattr(self, "sample_choice"):
			self.sample_choice.set(f"{row.word_id} | {row.word_text}")
		self._refresh_views()

	def _load_next_sample(self):
		if self.df_selected.empty:
			return

		current_path = self.current_path.get()
		matches = self.df_selected.index[self.df_selected["word_path"] == current_path].tolist()
		current_index = matches[0] if matches else 0
		next_index = (current_index + 1) % len(self.df_selected)
		self._load_row(next_index)

	def _on_sample_selected(self, _event):
		selection = self.sample_choice.get()
		if not selection:
			return

		word_id = selection.split(" | ", 1)[0]
		matches = self.df_selected.index[self.df_selected["word_id"] == word_id].tolist()
		if matches:
			self._load_row(matches[0])

	def _browse_file(self):
		path = filedialog.askopenfilename(initialdir=DATASET_WORDS_DIR, filetypes=[("PNG images", "*.png"), ("All files", "*.*")])
		if not path:
			return
		self.current_path.set(path)
		self.current_word.set(os.path.basename(path))
		self._refresh_views()

	def _load_current_image(self):
		path = self.current_path.get().strip()
		if not path:
			return None
		return cv2.imread(path)

	def _adaptive_method_value(self):
		return cv2.ADAPTIVE_THRESH_MEAN_C if self.adaptive_method.get() == "mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C

	def _threshold_type_value(self):
		return cv2.THRESH_BINARY_INV if self.invert_threshold.get() else cv2.THRESH_BINARY

	def _refresh_views(self):
		self.refresh_job = None
		image = self._load_current_image()
		if image is None:
			self.status_text.set("Nie udało się wczytać obrazu. Sprawdz sciezkę lub wybierz inny plik.")
			self.original_card.image_label.configure(image="")
			self.processed_card.image_label.configure(image="")
			self.original_card.meta_label.configure(text="")
			self.processed_card.meta_label.configure(text="")
			return

		self.current_image = image
		target_size = (max(1, int(self.target_width.get())), max(1, int(self.target_height.get())))
		processed = preprocess_handwriting_image(
			image,
			target_size=target_size,
			blur_kernel_size=int(self.blur_kernel_size.get()),
			threshold_block_size=int(self.threshold_block_size.get()),
			threshold_c=int(self.threshold_c.get()),
			adaptive_method=self._adaptive_method_value(),
			threshold_type=self._threshold_type_value(),
		)

		original_preview = resize_to_fit(image)

		processed_preview = resize_to_fit((processed * 255.0).clip(0, 255).astype(np.uint8))

		self.original_photo = image_to_photoimage(original_preview)
		self.processed_photo = image_to_photoimage(processed_preview)

		self.original_card.image_label.configure(image=self.original_photo)
		self.processed_card.image_label.configure(image=self.processed_photo)
		self.original_card.meta_label.configure(text=f"{image.shape[1]} x {image.shape[0]} px")
		self.processed_card.meta_label.configure(text=f"{processed.shape[1]} x {processed.shape[0]} px, target={target_size}")

		self.status_text.set(
			f"Plik: {self.current_path.get()} | target={target_size} | blur={int(self.blur_kernel_size.get())} | block={int(self.threshold_block_size.get())} | C={int(self.threshold_c.get())}"
		)


def main():
	root = tk.Tk()
	app = PreprocessingComparer(root)
	root.mainloop()


if __name__ == "__main__":
	main()