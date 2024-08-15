import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.io import read_image
from transformers import ViTForImageClassification


class ViTClf(nn.Module):
    def __init__(self, n_labels=2, model_path=None):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(
            model_path,
            num_labels=n_labels
        )

    def forward(self, batch):
        outputs = self.model(pixel_values=batch)
        logits = outputs.logits
        return logits
