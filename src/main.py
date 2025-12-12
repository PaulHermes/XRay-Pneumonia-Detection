import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import hyperparameters as hp
from src.data_management import (
  load_data,
  get_train_transforms,
  get_val_transforms,
  create_training_dataset_with_sampler,
  PneumoniaDataset,
)
from src.hyperparameters import (
  PRETRAINED, NUM_CLASSES,
  BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
)
from src.model import build_model

model_results = []

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, running_corrects = 0.0, 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # handle models that return {"logits": ...}
        if isinstance(outputs, dict):
            outputs = outputs["logits"]

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    dataset_size = len(dataloader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double().item() / dataset_size
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, running_corrects = 0.0, 0

    progress_bar = tqdm(dataloader, desc="Validating", leave=False)

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            if isinstance(outputs, dict):
                outputs = outputs["logits"]

            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    dataset_size = len(dataloader.dataset)
    val_loss = running_loss / dataset_size
    val_acc = running_corrects.double().item() / dataset_size
    return val_loss, val_acc


def train_and_save(model_save_path: Path, data_root: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(pretrained=PRETRAINED, num_classes=NUM_CLASSES).to(device)

    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()

    train_dataset, train_sampler = create_training_dataset_with_sampler(data_root)
    val_dataset = PneumoniaDataset(data_root, transform=val_transforms, split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_val_loss = float("inf")

    EARLY_STOPPING_PATIENCE = 3
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f" Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(model_save_path, exist_ok=True)
            save_path = model_save_path / f"best_model_{hp.MODEL_ARCH}_{hp.MODEL_SOURCE}.pth"
            torch.save(model.state_dict(), save_path)
            print(f" Saved new best model → {save_path}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early Stopping triggered after {epoch+1} epochs!")
            break

    model_results.append({
        "arch": hp.MODEL_ARCH,
        "source": hp.MODEL_SOURCE,
        "val_acc": best_val_acc,
        "val_loss": best_val_loss,
        "early_stopped": epochs_without_improvement >= EARLY_STOPPING_PATIENCE
    })

    print("\nTraining finished.")


def main():

    print("Downloading dataset via KaggleHub (if needed)...")
    data_info = load_data()
    data_root = data_info["data_root"]
    print(f"Dataset root is: {data_root}")

    model_save_path = Path("models/")
    model_save_path.mkdir(exist_ok=True)

    # ========== All model configurations to train ==========
    model_configs = [
        ("simple_cnn", "simple_cnn"),
        ("resnet50", "imagenet"),
    ]

    # ========== Train each model ==========
    for arch, source in model_configs:
        print("\n==============================================")
        print(f"   Training Model: {arch}  |  Source: {source}")
        print("==============================================")

        hp.MODEL_ARCH = arch
        hp.MODEL_SOURCE = source

        train_and_save(model_save_path, data_root)


    print("\n================ Final Model Results ================")
    print("Model".ljust(28), "| Val Acc | Val Loss | EarlyStop")
    print("-" * 58)

    for r in model_results:
        model_name = f"{r['arch']}/{r['source']}".ljust(28)
        acc = f"{r['val_acc']:.4f}"
        loss = f"{r['val_loss']:.4f}"
        es = "YES" if r["early_stopped"] else "NO"
        print(f"{model_name} | {acc}   | {loss}   | {es}")
    print("=" * 58)


if __name__ == "__main__":
    main()
