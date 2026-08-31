import matplotlib.pyplot as plt
import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history['loss'], label='Training Loss')
    if 'loss' in history:
        axes[0].plot(history['loss'], label='Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history['accuracy'], label='Training Accuracy')
    if 'accuracy' in history:
        axes[1].plot(history['accuracy'], label='Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training_history(load_data("training_siamese_20260825_175206/metrics.csv"))