import torch
import crnn
import siamese
import pandas as pd
import pre_trained_models

if __name__ == "__main__":
    encoder = pre_trained_models.ResNetEncoder(embedding_dim=128, pretrained=True)
    encoder.load_state_dict(torch.load("PretrainResnetClassifier_encoder_pretrained.pth"))
    siamese.train_siamese(pretrained_encoder=encoder, max_epochs = 60, resume_dir="training_siamese_SiameseNetwork_ResNetEncoder_AdamW_20260901_024151")