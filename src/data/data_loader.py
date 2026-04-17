import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from tqdm import tqdm


from src.utils.helpers import read_json


class PlayerNumberDataset(Dataset):
    def __init__(self, dataset_path, json_file, img_size):
        self.dataset_path = dataset_path
        self.samples = read_json(f"{dataset_path}/{json_file}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),                        # HWC uint8 → CHW float [0,1]
            transforms.Resize(img_size),
            transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean/std
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item      = self.samples[idx]
        raw_class = item["class"]

        # Load image (cv2 is BGR → convert to RGB)
        img_path   = f"{self.dataset_path}/bboxes/{item['relative_path']}"
        img        = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img)

        # Head-1 target  – class <= 0 means not legible
        legible = torch.tensor(1 if raw_class > 0 else 0, dtype=torch.long)

        # Head-2 target  – jersey number; 0 when not legible (ignored by loss)
        number  = torch.tensor(max(raw_class, 0), dtype=torch.long)

        return img_tensor, legible, number


def get_data_loaders(cfg):
    full_dataset = PlayerNumberDataset(cfg["dataset_path"], cfg["json_file"], cfg["img_size"])
    val_size   = int(len(full_dataset) * cfg["val_split"])
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],shuffle=True,  num_workers=cfg["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],shuffle=False, num_workers=cfg["num_workers"]) 

    print(f"Train samples: {train_size}  |  Val samples: {val_size}\n")

    return train_loader, val_loader