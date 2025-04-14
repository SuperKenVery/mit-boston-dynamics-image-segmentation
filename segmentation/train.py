#!/usr/bin/env python3

import os
import argparse
from typing import Dict, Any, Optional, Tuple, Union, Literal

from pytorch_lightning.utilities.types import OptimizerConfig, OptimizerLRSchedulerConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from segmentation.data import get_data_loader, get_object_classes
from segmentation.model import H, Segmenter


class SegmentationModel(pl.LightningModule):
    """PyTorch Lightning module for training the segmentation model."""

    def __init__(self,
                 num_classes: int = 40,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 model_variant: str = "19s"):
        super().__init__()
        self.save_hyperparameters()

        self.model = Segmenter(variant=model_variant)
        self.loss_fn = nn.CrossEntropyLoss()

        # Make sure the model outputs the right number of classes
        test_output = self.model(torch.zeros(1,3,64,64))
        assert test_output.shape[1] == num_classes, \
            f"Model outputs {test_output.shape[1]} classes (shape {test_output.shape}), expected {num_classes}"

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def _common_step(self, batch, batch_idx, stage):
        images, target = batch
        B, C, H, W = images.shape

        # Forward pass
        outputs = self(images)  # Shape: [B, Classes, H, W]

        # Calculate loss
        loss = self.loss_fn(outputs, target)

        # Calculate accuracy
        with torch.no_grad():
            pred_indices = torch.argmax(outputs, dim=1)  # [B, H, W]
            image_classification = torch.mode(outputs.reshape(B, H * W), dim=1)[0]   # [B]
            gt_classification = torch.mode(target.reshape(B, H * W), dim=1)[0]   # [B, H, W] -> [B]
            correct = (image_classification == gt_classification).float().sum()
            accuracy = correct / B

        # Log metrics
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_accuracy", accuracy, prog_bar=True)

        return {"loss": loss, "accuracy": accuracy}

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )


        return OptimizerLRSchedulerConfig({
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        })


def train(
    data_dir: str = "data/training",
    batch_size: int = 8,
    num_workers: int = 4,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    max_epochs: int = 100,
    model_variant: str = "19s",
    gpus: int = 1,
    precision: Union[Literal[16], Literal[32], Literal[64]] = 32,
    checkpoint_dir: str = "checkpoints",
    patience: int = 10,
):
    """Train the segmentation model.

    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        max_epochs: Maximum number of epochs to train
        model_variant: VGG variant to use
        gpus: Number of GPUs to use for training
        precision: Precision for training (16, 32, or 64)
        checkpoint_dir: Directory to save checkpoints
        patience: Number of epochs to wait before early stopping
    """
    pl.seed_everything(42)

    # Get the number of object classes
    object_classes = get_object_classes(data_dir)
    # num_classes = len(object_classes) + 1  # +1 for background class
    num_classes = 40

    # Create data loaders
    train_loader = get_data_loader(
        root_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    val_loader = get_data_loader(
        root_dir=data_dir,  # Ideally use a separate validation set
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    # Create model
    model = SegmentationModel(
        num_classes=num_classes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        model_variant=model_variant,
    )

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="segmentation-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=patience,
        verbose=True,
        mode="min",
    )

    # Create logger
    logger = TensorBoardLogger("lightning_logs", name="segmentation")

    # Determine accelerator
    accelerator = None
    if torch.backends.mps.is_available():
        accelerator = "mps"
    elif torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=gpus,
        precision=precision,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    return model, trainer


def main():
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to the dataset")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for the optimizer")
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum number of epochs to train")
    parser.add_argument("--model-variant", type=str, default="19s", help="VGG variant to use")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32, 64], help="Precision for training")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--patience", type=int, default=10, help="Number of epochs to wait before early stopping")

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        model_variant=args.model_variant,
        gpus=args.gpus,
        precision=args.precision,
        checkpoint_dir=args.checkpoint_dir,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
