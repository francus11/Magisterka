import torch.nn as nn
from torchvision.models import resnet18


class ResNetEncoder(nn.Module):
	"""Enkoder ResNet zwracający wektor cech kompatybilny z SiameseNetwork."""

	def __init__(self, embedding_dim=128, pretrained=False):
		super().__init__()

		backbone = resnet18(weights="DEFAULT" if pretrained else None)

		first_layer = backbone.conv1
		backbone.conv1 = nn.Conv2d(
			in_channels=1,
			out_channels=first_layer.out_channels,
			kernel_size=first_layer.kernel_size,
			stride=first_layer.stride,
			padding=first_layer.padding,
			bias=False,
		)

		if pretrained:
			backbone.conv1.weight.data.copy_(first_layer.weight.data.mean(dim=1, keepdim=True))

		backbone.fc = nn.Linear(backbone.fc.in_features, embedding_dim)
		self.backbone = backbone

	def forward(self, x):
		return self.backbone(x)
