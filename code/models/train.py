import logging
import os
import warnings

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")

# --- LOGGER --- #

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


# --- DEVICE --- #


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple GPU
    else:
        return "cpu"


device = get_device()
logger.info(f"Using device: {device}")

# --- CONFIG --- #

CONFIG = {
    "img_path": "../datasets/images",
    "csv_path": "../datasets/labels.csv",
    "batch_size": 64,
    "epochs": 10,
    "lr": 0.001,
    "ckpt_path": "../../models/model.pt",
    "device": device,
}


# --- DATASET --- #


class CelebADataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx]["image_id"])
        image = read_image(img_path) / 255.0
        label = self.img_labels.iloc[idx]["Blond_Hair"]
        label = 0 if label == -1 else label
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


data_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
    ]
)


def get_dataloaders(csv_path, img_dir, batch_size, test_size=0.2, random_state=42):
    dataset = CelebADataset(
        csv_file=csv_path, img_dir=img_dir, transform=data_transforms
    )
    train_data, val_data = train_test_split(
        dataset,
        test_size=test_size,
        stratify=dataset.img_labels["Blond_Hair"],
        random_state=random_state,
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


train_loader, val_loader = get_dataloaders(
    CONFIG["csv_path"], CONFIG["img_path"], CONFIG["batch_size"]
)


# --- MODEL --- #


class CNNClassificationModel(nn.Module):
    def __init__(self, num_classes=1):
        super(CNNClassificationModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=3, stride=1, padding=1
            ),  # [3, 64, 64] -> [32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [32, 64, 64] -> [32, 32, 32]
            nn.Conv2d(
                32, 64, kernel_size=3, stride=1, padding=1
            ),  # [32, 32, 32] -> [64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [64, 32, 32] -> [64, 16, 16]
            nn.Conv2d(
                64, 128, kernel_size=3, stride=1, padding=1
            ),  # [64, 16, 16] -> [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [128, 16, 16] -> [128, 8, 8]
            nn.Conv2d(
                128, 256, kernel_size=3, stride=1, padding=1
            ),  # [128, 8, 8] -> [256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [256, 8, 8] -> [256, 4, 4]
            nn.Flatten(),  # [256, 4, 4] -> [4096]
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)


model = CNNClassificationModel().to(CONFIG["device"])


# --- TRAINING --- #


def train_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    epoch_loss = running_loss / total_train
    accuracy = correct_train / total_train
    return epoch_loss, accuracy


def validate_epoch(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            predicted = (outputs > 0.5).float()
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    epoch_loss = val_loss / total_val
    accuracy = correct_val / total_val
    return epoch_loss, accuracy


def train_model(
    model, train_loader, val_loader, optimizer, loss_fn, epochs, device, ckpt_path
):
    best_val_loss = float("inf")

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        val_loss, val_acc = validate_epoch(model, val_loader, loss_fn, device)

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            logger.info(
                f"Validation loss improved ({best_val_loss:.4f} -> {val_loss:.4f}), saving model..."
            )
            torch.save(model.state_dict(), ckpt_path)
            best_val_loss = val_loss


optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
loss_fn = nn.BCEWithLogitsLoss()

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=CONFIG["epochs"],
    device=CONFIG["device"],
    ckpt_path=CONFIG["ckpt_path"],
)
