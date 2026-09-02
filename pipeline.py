import torch
from siamese import train_siamese
from pre_trained_models import ResNetEncoder
from crnn import train_pretrain
from crnn import PretrainResnetClassifier
import crnn

def siamese_training_ResNetEncoder():
    pretrained_encoder = ResNetEncoder(embedding_dim=128, pretrained=True)
    pretrained_encoder.load_state_dict(torch.load("PretrainResnetClassifier_encoder_pretrained.pth"))
    train_siamese(pretrained_encoder=pretrained_encoder, max_epochs=30, samples_per_epoch=20000, batch_size=8, learning_rate=1e-3, margin=1.0, threshold=0.5)

def classifier_training_ResNetEncoder_WithDropout():
    model = crnn.PretrainResnetClassifier_WithDropout
    train_pretrain(max_epochs=30, batch_size=8, learning_rate=1e-3, model=model, resume_dir='training_crnn_PretrainResnetClassifier_WithDropout_ResNetEncoder_Adam_20260901_070459')

def siamese_training_ResNetEncoder_WithDropout():
    pretrained_encoder = crnn.ResNetEncoder(embedding_dim=128, pretrained=True, dropout_p=0.5)
    pretrained_encoder.load_state_dict(torch.load("training_crnn_PretrainResnetClassifier_WithDropout_ResNetEncoder_Adam_20260901_070459/PretrainResnetClassifier_WithDropout_encoder_pretrained.pth"))
    train_siamese(pretrained_encoder=pretrained_encoder, max_epochs=30, samples_per_epoch=20000, batch_size=8, learning_rate=1e-3, margin=1.0, threshold=0.5)

if __name__ == "__main__":
    siamese_training_ResNetEncoder_WithDropout()
    # Example usage of the imported modules
    # siamese_training_ResNetEncoder()
    # model = PretrainResnetClassifier
    # train_pretrain(max_epochs=30, batch_size=8, learning_rate=1e-3, model=model)

    