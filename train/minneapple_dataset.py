#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Tuple, Union

from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.v2 as T
from collections.abc import Iterable
from torch.utils.data import WeightedRandomSampler


class MinneAppleDataset(Dataset):
    def __init__(self,
                 annotations_file: Union[str, Path],
                 classes: Tuple = (0, 1),
                 transform=None):
        self.annotations = []
        self.images_path = Path(annotations_file).parent / 'images'
        df = pd.read_csv(annotations_file)

        def map_label(num_apples):
            for c_idx, c in enumerate(classes):
                if num_apples == c or (isinstance(c, Iterable) and num_apples in c):
                    return c_idx
            return None

        df['label'] = df['count'].apply(map_label)
        df = df.dropna()
        df['img_path'] = df['Image'].apply(lambda x: str(self.images_path / x))
        self.n_neg = len(df[df['label'].astype(int) == 0])
        self.n_pos = len(df[df['label'].astype(int) == 1])
        self.annotations = list(zip(df['img_path'], df['label'].astype(int)))
        self.labels = torch.tensor([label for _, label in self.annotations])
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path, label = self.annotations[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_rates(self):
        return self.n_neg, self.n_pos


def get_train_transforms(input_size=224):
    return T.Compose([T.Resize((input_size, input_size)),
                      T.RandomHorizontalFlip(0.5),
                      T.RandomVerticalFlip(0.5),
                      T.RandomRotation(20),
                      T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                      T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                      T.ToImage(), T.ToDtype(torch.float32, scale=True),
                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def get_val_transforms(input_size=224):
    return T.Compose([T.Resize((input_size, input_size)),
                      T.ToImage(), T.ToDtype(torch.float32, scale=True),
                      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def get_dataloaders(train_dataset: MinneAppleDataset,
                    val_dataset: MinneAppleDataset,
                    batch_size=32,
                    num_workers=4):
    class_counts = [train_dataset.n_neg, train_dataset.n_pos]
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              # shuffle=True,
                              num_workers=num_workers,
                              sampler=sampler)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    return train_loader, val_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = MinneAppleDataset('../../datasets/minneapple/counting/train/train_ground_truth.txt',
                                classes=((0, 3, 4, 5), 1),
                                transform=get_val_transforms(100))
    print(f'Dataset size: {len(dataset)}')

    train_loader, val_loader = get_dataloaders(dataset,
                                               dataset,
                                               batch_size=36)

    batch, labels = next(iter(train_loader))
    fig = plt.figure(figsize=(10, 10))
    for i in range(36):
        plt.subplot(6, 6, i + 1)
        plt.title(labels[i].item())
        img = batch[i].permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img.numpy())
        plt.axis('off')
    plt.show()
