import os
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, make_scorer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import hyperparameters as hp
from src.data_management import (load_data, get_train_transforms, get_val_transforms, create_training_dataset_with_sampler, PneumoniaDataset)
from src.evaluation import plot_learning_curve_for_model, plot_training_history, generate_cam_visualizations, calculate_metrics, evaluate_on_test_set
from src.hyperparameters import (PRETRAINED, NUM_CLASSES, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS)
from src.model import build_model

model_results = []

def train_one_epoch(model, dataloader, criterion, optimizer, device):
  model.train()
  running_loss = 0.0
  all_preds, all_labels = [], []

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
    
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

  epoch_loss = running_loss / len(dataloader.dataset)
  metrics = calculate_metrics(all_labels, all_preds)
  metrics["loss"] = epoch_loss
  return metrics


def validate(model, dataloader, criterion, device):
  model.eval()
  running_loss = 0.0
  all_preds, all_labels = [], []
  
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
      all_preds.extend(preds.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())
      progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

  epoch_loss = running_loss / len(dataloader.dataset)
  metrics = calculate_metrics(all_labels, all_preds)
  metrics["loss"] = epoch_loss
  return metrics


def train_and_save(model_save_path: Path, data_root: Path):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  model = build_model(pretrained=PRETRAINED, num_classes=NUM_CLASSES).to(device)

  train_dataset, train_sampler = create_training_dataset_with_sampler(data_root)
  val_dataset = PneumoniaDataset(data_root, transform=get_val_transforms(), split="val")

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
  val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(
      model.parameters(), 
      lr=LEARNING_RATE,
      weight_decay=hp.optimizer_params["weight_decay"]
  )
  
  scheduler = None
  if hp.SCHEDULER_CONFIG["enabled"]:
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
          optimizer, 
          mode=hp.SCHEDULER_CONFIG["mode"], 
          factor=hp.SCHEDULER_CONFIG["factor"], 
          patience=hp.SCHEDULER_CONFIG["patience"], 
          verbose=True
      )

  best_metric_value = -float('inf') if hp.MONITOR_MODE == 'max' else float('inf')
  best_early_stopping_metric_value = -float('inf') if hp.MONITOR_MODE == 'max' else float('inf')
  
  history = defaultdict(list)

  epochs_without_improvement = 0
  early_stopped = False

  for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

    train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_metrics = validate(model, val_loader, criterion, device)

    if scheduler:
        scheduler.step(val_metrics[hp.SCHEDULER_CONFIG["monitor"]])

    # Store metrics
    for key, value in train_metrics.items():
        history[f"train_{key}"].append(value)
    for key, value in val_metrics.items():
        history[f"val_{key}"].append(value)

    print(f" Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
    print(f" Val   Loss: {val_metrics['loss']:.4f} | Val   Acc: {val_metrics['accuracy']:.4f}")

    # Save best model
    current_metric = val_metrics[hp.MONITOR_METRIC]
    if (hp.MONITOR_MODE == 'max' and current_metric > best_metric_value) or \
       (hp.MONITOR_MODE == 'min' and current_metric < best_metric_value):
        best_metric_value = current_metric
        save_path = model_save_path / f"best_model_{hp.MODEL_ARCH}_{hp.MODEL_SOURCE}.pth"
        torch.save(model.state_dict(), save_path)
        print(f" Saved new best model → {save_path} (best {hp.MONITOR_METRIC}: {best_metric_value:.4f})")

    # Early Stopping
    if hp.EARLY_STOPPING_CONFIG.get("enabled", False):
        if (hp.MONITOR_MODE == 'max' and current_metric > best_early_stopping_metric_value) or \
           (hp.MONITOR_MODE == 'min' and current_metric < best_early_stopping_metric_value):
            best_early_stopping_metric_value = current_metric
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= hp.EARLY_STOPPING_CONFIG["patience"]:
            print(f"Early Stopping triggered after {epoch + 1} epochs!")
            early_stopped = True
            break
  
  if not hp.EARLY_STOPPING_CONFIG.get("enabled", False):
      early_stopped = False
  else:
      early_stopped = epochs_without_improvement >= hp.EARLY_STOPPING_CONFIG["patience"]

  model_results.append({
    "arch": hp.MODEL_ARCH,
    "source": hp.MODEL_SOURCE,
    "val_acc": val_metrics['accuracy'],
    "val_loss": val_metrics['loss'],
    "early_stopped": early_stopped
  })

  print("\nTraining finished.")
  
  plot_training_history(history, hp.MODEL_ARCH, hp.MODEL_SOURCE)


def main():
  print("Downloading dataset via KaggleHub (if needed)...")
  data_info = load_data()
  data_root = data_info["data_root"]
  print(f"Dataset root is: {data_root}")

  model_save_path = Path("models/")
  model_save_path.mkdir(exist_ok=True)
  
  plots_path = Path("plots/")
  plots_path.mkdir(exist_ok=True)


  # ========== All model configurations to train ==========
  model_configs = []
  if hp.TRAIN_SIMPLE_CNN:
    model_configs.append(("simple_cnn", "simple_cnn"))
  if hp.TRAIN_RESNET:
    model_configs.append(("resnet50", "imagenet"))

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
  
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # ========== Generate Learning Curves ==========
  if hp.PLOT_LEARNING_CURVES:
    print("\n==============================================")
    print("      Generating Learning Curves")
    print("==============================================")
    
    # Define the metrics to plot
    specificity_scorer = make_scorer(recall_score, pos_label=0)
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1_score': 'f1_macro',
        'specificity': specificity_scorer
    }

    for arch, source in model_configs:
      plot_learning_curve_for_model(arch, source, data_root, device, scoring_metrics)

  # ========== Generate CAM Visualizations ==========
  if hp.CAM.get("USE_CAM", False):
    print("\n==============================================")
    print("      Generating CAM Visualizations")
    print("==============================================")
    for arch, source in model_configs:
      generate_cam_visualizations(arch, source, data_root, device)

  # ========== Test Set Evaluation ==========
  print("\n==============================================")
  print("      Test Set Evaluation")
  print("==============================================")
  for arch, source in model_configs:
    evaluate_on_test_set(arch, source, data_root, device)

if __name__ == "__main__":
  main()
