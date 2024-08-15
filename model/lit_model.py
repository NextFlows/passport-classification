import torch
import torch.optim as optim
import torchmetrics
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from transformers import AdamW

from model.vit import ViTClf


class LitClassifier(LightningModule):
    def __init__(self, num_labels=4, learning_rate=5e-5, model_path=None):
        super().__init__()
        self.lr = learning_rate
        self.num_labels = num_labels
        self.save_hyperparameters()

        self.model = ViTClf(n_labels=self.num_labels, model_path=model_path)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.train_accuracy = torchmetrics.Accuracy('binary')
        self.val_accuracy = torchmetrics.Accuracy('binary')
        self.train_f1 = torchmetrics.F1Score(num_classes=self.num_labels, average="macro")
        self.val_f1 = torchmetrics.F1Score(num_classes=self.num_labels, average="macro")

    def forward(self, x):
        x = self.model(x)
        # Convert the 4-d id classifier to non-passport and passport
        # x = torch.tensor([torch.sum(x[0][:3]), x[0][3]])
        return x

    def common_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.common_step(batch, batch_idx)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.train_f1(preds, y)

        self.log("train_loss", loss)
        self.log("train_acc", self.train_accuracy, prog_bar=True, on_epoch=True)
        self.log("train_f1", self.train_f1, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.common_step(batch, batch_idx)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)
        self.val_f1(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_f1", self.val_f1, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR
        # optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
