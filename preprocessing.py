import os
import shutil

import cv2
import numpy as np
import pandas as pd


def _normalize_odd_kernel_size(value, minimum=3):
	kernel_size = max(minimum, int(value))
	if kernel_size % 2 == 0:
		kernel_size += 1
	return kernel_size


def preprocess_handwriting_image(
	image,
	target_size=(128, 128),
	blur_kernel_size=3,
	threshold_block_size=31,
	threshold_c=11,
	adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	threshold_type=cv2.THRESH_BINARY_INV,
	resize_interpolation=cv2.INTER_AREA,
):
	if image is None:
		return None

	img = image.copy()
	if img.ndim == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	if target_size is not None:
		if isinstance(target_size, (tuple, list)) and len(target_size) == 2 and 0 in target_size:
			current_height, current_width = img.shape[:2]
			target_width, target_height = target_size
			if target_width == 0 and target_height == 0:
				target_size = None
			elif target_width == 0:
				scale = target_height / float(current_height)
				target_size = (max(1, int(round(current_width * scale))), target_height)
			elif target_height == 0:
				scale = target_width / float(current_width)
				target_size = (target_width, max(1, int(round(current_height * scale))))
		img = cv2.resize(img, target_size, interpolation=resize_interpolation)

	blur_kernel_size = _normalize_odd_kernel_size(blur_kernel_size, minimum=1)
	if blur_kernel_size > 1:
		img = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)

	threshold_block_size = _normalize_odd_kernel_size(threshold_block_size, minimum=3)
	img = cv2.adaptiveThreshold(
		img,
		255,
		adaptive_method,
		threshold_type,
		threshold_block_size,
		threshold_c,
	)
	return img.astype(np.float32) / 255.0


def load_and_preprocess_image(
	path,
	target_size=(128, 128),
	blur_kernel_size=3,
	threshold_block_size=31,
	threshold_c=11,
	adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	threshold_type=cv2.THRESH_BINARY_INV,
	resize_interpolation=cv2.INTER_AREA,
	minimal_size=(1, 1)
):
	if not isinstance(path, str) or not os.path.exists(path):
		return None

	image = cv2.imread(path)
	if image is None:
		return None

	if target_size is None:
		target_size = image.shape[1], image.shape[0]

	return preprocess_handwriting_image(
		image,
		target_size=target_size,
		blur_kernel_size=blur_kernel_size,
		threshold_block_size=threshold_block_size,
		threshold_c=threshold_c,
		adaptive_method=adaptive_method,
		threshold_type=threshold_type,
		resize_interpolation=resize_interpolation,
	)


def build_word_dataframe(df, pics_path):
	df_selected = df[["word_id", "word_text", "user_class", "word_bboxes"]].copy()
	df_selected = df_selected.explode(["word_id", "word_text", "word_bboxes"], ignore_index=True)
	df_selected["word_path"] = pics_path + "/" + df_selected["word_id"].astype(str) + ".png"


	def to_flat_bbox(x):
		if isinstance(x, np.ndarray):
			if x.ndim == 1 and x.shape[0] == 4:
				return x.astype(float).tolist()
			return None
		if isinstance(x, (list, tuple)) and len(x) == 4 and all(np.isscalar(v) for v in x):
			return [float(v) for v in x]
		return None

	df_selected = df_selected[df_selected["word_bboxes"].notna()].copy()

	coords = pd.DataFrame(
		df_selected["word_bboxes"].tolist(),
		columns=["x1", "y1", "x2", "y2"],
		index=df_selected.index,
	)
	df_selected[["x1", "y1", "x2", "y2"]] = coords

	df_selected["bbox_width"] = df_selected["x2"] - df_selected["x1"]
	df_selected["bbox_height"] = df_selected["y2"] - df_selected["y1"]

	df_selected.drop(columns=["x1", "y1", "x2", "y2"], inplace=True)

	return df_selected

def filter_records_valid_height(df_selected, min_height=1):
	return df_selected[df_selected["bbox_height"] >= min_height].copy().reset_index(drop=True)

def export_preprocessed_images(df_selected, output_dir="preprocessed_words", **preprocess_kwargs):
	if os.path.isdir(output_dir):
		shutil.rmtree(output_dir)
	os.makedirs(output_dir, exist_ok=True)
	preprocessed_images = df_selected["word_path"].apply(load_and_preprocess_image, **preprocess_kwargs)

	
	for idx, img in preprocessed_images.items():
		if img is None:
			continue

		out = (img * 255.0).clip(0, 255).astype(np.uint8)
		if out.ndim == 3 and out.shape[2] == 1:
			out = out.squeeze(2)

		original_filename = os.path.basename(df_selected.at[idx, "word_path"])
		out_path = os.path.join(output_dir, original_filename)
		cv2.imwrite(out_path, out)

	return preprocessed_images


def main():
	print("Starting preprocessing...")

	df = pd.read_parquet("df.parquet")
	df_selected = build_word_dataframe(df, pics_path="dataset_words/words")

	df_selected = filter_records_valid_height(df_selected, min_height=32)

	preprocessed_images = export_preprocessed_images(df_selected, target_size=None)

	df_selected.to_parquet("df_words_preprocessed.parquet", index=False)

	print("Preprocessing completed. Preprocessed images saved in 'preprocessed_words' directory.")

if __name__ == "__main__":
	main()


