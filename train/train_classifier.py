#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import hydra
from omegaconf import DictConfig

from train.minneapple_dataset import get_dataloaders, MinneAppleDataset, get_train_transforms, get_val_transforms


def train_epoch(model,
                dataloader,
                criterion,
                optimizer,
                device,
                epoch):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1} [Train]')

    for batch_idx, (data, targets) in enumerate(progress_bar):
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += targets.size(0)
        correct_predictions += (predicted == targets).sum().item()

        avg_loss = running_loss / (batch_idx + 1)
        avg_acc = correct_predictions / total_samples

        progress_bar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{avg_acc:.4f}'
        })

    avg_loss = running_loss / len(dataloader)
    avg_acc = correct_predictions / total_samples

    mlflow.log_metric('train_loss', avg_loss, step=epoch)
    mlflow.log_metric('train_accuracy', avg_acc, step=epoch)

    return avg_loss, avg_acc


def evaluate(model, dataloader, criterion, device, epoch=-1):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1} [Validation]' if epoch != -1 else 'Validation')

        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(device), targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()

            avg_loss = running_loss / (batch_idx + 1)
            avg_acc = correct_predictions / total_samples

            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{avg_acc:.4f}'
            })

    avg_loss = running_loss / len(dataloader)
    avg_acc = correct_predictions / total_samples

    if epoch != -1:
        mlflow.log_metric('val_loss', avg_loss, step=epoch)
        mlflow.log_metric('val_accuracy', avg_acc, step=epoch)

    return avg_loss, avg_acc


def train_model(model,
                train_loader,
                val_loader,
                num_epochs,
                learning_rate,
                device,
                save_path=None):


    with mlflow.start_run():  # ← wrap the whole training loop
        mlflow.log_params({"num_epochs": num_epochs,
                           "learning_rate": learning_rate,
                           "batch_size": train_loader.batch_size,
                           "model": type(model).__name__})
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []

        best_val_acc = 0.0

        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if best_val_acc > 0.8:
                    mlflow.pytorch.log_model(
                        pytorch_model=model,
                        artifact_path="best_model",
                        code_paths=["train/train_classifier.py", "train/minneapple_dataset.py"]
                    )
                    if save_path:
                        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                        torch.save({'epoch': epoch,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'val_acc': val_acc},
                                   save_path)
                        print(f'Best model saved with accuracy: {best_val_acc:.4f}')

    return train_losses, train_accuracies, val_losses, val_accuracies


def test_model(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    return test_loss, test_acc


@hydra.main(config_path='../configs/train', config_name='config', version_base=None)
def main(cfg: DictConfig):
    print(Path().absolute())
    if cfg.mlflow.tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    model = getattr(torchvision.models, cfg.model.name)(pretrained=cfg.model.pretrained)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=cfg.model.num_classes)

    train_set = MinneAppleDataset(cfg.dataset.train_annotations,
                                  classes=cfg.dataset.classes,
                                  transform=get_train_transforms(cfg.dataset.input_size))
    val_set = MinneAppleDataset(cfg.dataset.val_annotations,
                                classes=cfg.dataset.classes,
                                transform=get_val_transforms(cfg.dataset.input_size))
    print(f'train size: {len(train_set)}, val size: {len(val_set)}')

    train_loader, val_loader = get_dataloaders(train_set, val_set,
                                               batch_size=cfg.training.batch_size,
                                               num_workers=cfg.training.num_workers)

    device = torch.device(cfg.device)
    train_model(model,
                train_loader,
                val_loader,
                num_epochs=cfg.training.num_epochs,
                learning_rate=cfg.training.learning_rate,
                device=device,
                save_path=cfg.training.save_path)


if __name__ == '__main__':
    main()