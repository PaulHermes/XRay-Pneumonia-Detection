import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import learning_curve, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, recall_score, confusion_matrix, accuracy_score, precision_score, f1_score
from skorch import NeuralNetClassifier
from collections import defaultdict
import os
import seaborn as sns
import json
from torch.utils.data import DataLoader

from src import hyperparameters as hp
from src.model import build_model
from src.data_management import PneumoniaDataset, get_train_transforms, get_val_transforms, load_data, get_weighted_sampler
from src.cam import generate_cam_image

def plot_learning_curve_for_model(model_arch, model_source, data_root, device, scoring_metrics):
    print(f"Generating learning curves for {model_arch}/{model_source}...")

    # 1. Setup
    hp.MODEL_ARCH = model_arch
    hp.MODEL_SOURCE = model_source
    model_class = lambda: build_model(pretrained=hp.PRETRAINED, num_classes=hp.NUM_CLASSES)

    net = NeuralNetClassifier(
        module=model_class,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=hp.LEARNING_RATE,
        batch_size=hp.BATCH_SIZE,
        train_split=None,
        verbose=0,
        device=device,
    )

    # 2. Load data
    train_transforms = get_train_transforms()
    train_dataset = PneumoniaDataset(data_root, transform=train_transforms, split='train')
    
    X_all = np.array([item[0].numpy() for item in train_dataset])
    y_all = np.array([item[1] for item in train_dataset])
    
    sampler = get_weighted_sampler(train_dataset)

    # 3. Define training sizes and results storage
    train_sizes_abs = np.linspace(int(0.1 * len(X_all)), len(X_all), 5, dtype=int)
    results = defaultdict(lambda: defaultdict(list))
    
    # 4. Iterate over training sizes and compute all metrics at once
    for i, n_samples in enumerate(train_sizes_abs):
        print(f"  - [{i+1}/{len(train_sizes_abs)}] Training on {n_samples} samples...")
        
        # Generate a subset of indices using the weighted sampler
        sampler_iter = iter(sampler)
        subset_indices = [next(sampler_iter) for _ in range(n_samples)]
        
        X_subset, y_subset = X_all[subset_indices], y_all[subset_indices]

        cv = StratifiedKFold(n_splits=3)

        cv_results = cross_validate(
            estimator=net,
            X=X_subset,
            y=y_subset,
            cv=cv,
            scoring=scoring_metrics,
            return_train_score=True,
            n_jobs=1
        )

        for metric in scoring_metrics:
            results[metric]['train_mean'].append(np.mean(cv_results[f"train_{metric}"]))
            results[metric]['train_std'].append(np.std(cv_results[f"train_{metric}"]))
            results[metric]['test_mean'].append(np.mean(cv_results[f"test_{metric}"]))
            results[metric]['test_std'].append(np.std(cv_results[f"test_{metric}"]))

    # 5. Plot results for each metric
    print("\nFinished calculations. Generating plots...")
    actual_train_sizes = [len(r['train_mean']) for r in results.values()][0]
    train_sizes_proportions = train_sizes_abs[:actual_train_sizes] / len(X_all)

    for metric_name, data in results.items():
        plt.figure(figsize=(10, 6))
        
        train_mean = np.array(data['train_mean'])
        train_std = np.array(data['train_std'])
        test_mean = np.array(data['test_mean'])
        test_std = np.array(data['test_std'])

        plt.plot(train_sizes_proportions, train_mean, 'o-', color='r', label='Training score')
        plt.fill_between(train_sizes_proportions, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
        
        plt.plot(train_sizes_proportions, test_mean, 'o-', color='g', label='Cross-validation score')
        plt.fill_between(train_sizes_proportions, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

        plt.title(f'Learning Curve for {model_arch} ({model_source}) - {metric_name.capitalize()}')
        plt.xlabel('Fraction of Training Set Size')
        plt.ylabel(metric_name.capitalize())
        plt.legend(loc='best')
        plt.grid()
        
        plot_path = f'plots/learning_curve_{model_arch}_{model_source}_{metric_name}.png'
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"  - Plot saved to {plot_path}")
        plt.close()


def plot_training_history(history, model_arch, model_source):
    print(f"\nPlotting training history for {model_arch}/{model_source}...")
    
    metric_keys = sorted(list(set(key.replace('train_', '').replace('val_', '') for key in history.keys())))
    
    epochs = range(1, len(history['train_loss']) + 1)

    for key in metric_keys:
        plt.figure(figsize=(10, 6))
        
        plt.plot(epochs, history[f"train_{key}"], 'o-', label=f'Training {key.capitalize()}')
        plt.plot(epochs, history[f"val_{key}"], 'o-', label=f'Validation {key.capitalize()}')
        
        plt.title(f'Training and Validation {key.capitalize()} for {model_arch} ({model_source})')
        plt.xlabel('Epochs')
        plt.ylabel(key.capitalize())
        plt.legend(loc='best')
        plt.grid()
        
        plot_path = f'plots/history_{model_arch}_{model_source}_{key}.png'
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"  - History plot for '{key}' saved to {plot_path}")
        plt.close()

def generate_cam_visualizations(model_arch, model_source, data_root, device):
    print(f"\nGenerating CAM visualizations for {model_arch}/{model_source}...")

    # 1. Setup output directory
    output_dir = hp.CAM["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)

    # 2. Build and load the best model
    hp.MODEL_ARCH = model_arch
    hp.MODEL_SOURCE = model_source
    model = build_model(pretrained=False, num_classes=hp.NUM_CLASSES) # pretrained=False because we load local weights
    model_path = f'models/best_model_{model_arch}_{model_source}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Get validation data
    val_transforms = get_val_transforms()
    val_dataset = PneumoniaDataset(data_root, transform=val_transforms, split='val')
    
    # 4. Get target layers
    target_layer_names = hp.CAM["TARGET_LAYER_MAP"].get(model_arch)
    if not target_layer_names:
        print(f"Warning: No target layer specified for model '{model_arch}' in hyperparameters. Skipping CAM generation.")
        return
        
    # Ensure it's a list
    if isinstance(target_layer_names, str):
        target_layer_names = [target_layer_names]

    # 5. Generate and save CAM images
    num_images_per_class = hp.CAM["NUM_IMAGES"] // 2
    
    class_indices = {0: [], 1: []}
    for i in range(len(val_dataset)):
        _, label = val_dataset[i]
        if len(class_indices[label]) < num_images_per_class:
            class_indices[label].append(i)
        if all(len(v) == num_images_per_class for v in class_indices.values()):
            break
            
    image_indices = class_indices[0] + class_indices[1]
    for layer_name_str in target_layer_names:
        try:
            target_layer = model
            for name in layer_name_str.split('.'):
                if name.endswith(']'):
                    name, index = name[:-1].split('[')
                    target_layer = getattr(target_layer, name)
                    target_layer = target_layer[int(index)]
                else:
                    target_layer = getattr(target_layer, name)
        except (AttributeError, IndexError) as e:
            print(f"Error: Could not find target layer '{layer_name_str}' in model '{model_arch}': {e}. Skipping.")
            continue
            
        print(f"  - Generating CAMs for layer: {layer_name_str}")

        for i, img_idx in enumerate(image_indices):
            input_tensor, label = val_dataset[img_idx]
            input_tensor = input_tensor.to(device)

            overlay, pred_idx = generate_cam_image(model, target_layer, input_tensor)

            actual_label = "Pneumonia" if label == 1 else "Normal"
            pred_label = "Pneumonia" if pred_idx == 1 else "Normal"
            
            sanitized_layer = layer_name_str.replace('[', '_').replace(']', '').replace('.', '_')
            
            filename = f"{model_arch}_{model_source}_{sanitized_layer}_img{i}_actual-{actual_label}_pred-{pred_label}.png"
            save_path = os.path.join(output_dir, filename)
            overlay.save(save_path)

def calculate_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    specificity = recall_score(labels, preds, pos_label=0, average='binary', zero_division=0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity
    }

def plot_confusion_matrix(y_true, y_pred, model_arch, model_source, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_arch} ({model_source})')
    plot_path = os.path.join(output_dir, f'confusion_matrix_{model_arch}_{model_source}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"  - Confusion matrix saved to {plot_path}")

def evaluate_on_test_set(model_arch, model_source, data_root, device):
    print(f"\nEvaluating on test set for {model_arch}/{model_source}...")

    # 1. Setup output directory
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Build and load the best model
    hp.MODEL_ARCH = model_arch
    hp.MODEL_SOURCE = model_source
    model = build_model(pretrained=False, num_classes=hp.NUM_CLASSES)
    model_path = f'models/best_model_{model_arch}_{model_source}.pth'
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        if "features" in str(e) and "conv" in str(e):
            print("  - Detected state_dict key mismatch. Trying to load from converted model...")
            model_path = f'models/best_model_{model_arch}_{model_source}_converted.pth'
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
            except FileNotFoundError:
                print(f"  - Error: Converted model not found at {model_path}.")
                print("  - Please run the conversion script first.")
                return
        else:
            raise e
            
    model.to(device)
    model.eval()

    # 3. Get test data
    test_transforms = get_val_transforms() # Use validation transforms for test set
    test_dataset = PneumoniaDataset(data_root, transform=test_transforms, split='test')
    test_loader = DataLoader(test_dataset, batch_size=hp.BATCH_SIZE, shuffle=False)

    # 4. Get predictions
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, dict):
                outputs = outputs["logits"]
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Calculate and save metrics
    metrics = calculate_metrics(all_labels, all_preds)
    print("  - Test Metrics:")
    for key, value in metrics.items():
        print(f"    - {key.capitalize()}: {value:.4f}")

    # 6. Save metrics to a file
    report = {
        "model": f"{model_arch}_{model_source}",
        "metrics": metrics
    }
    report_path = os.path.join(output_dir, f'test_report_{model_arch}_{model_source}.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"  - Test report saved to {report_path}")

    # 7. Plot and save confusion matrix
    plot_confusion_matrix(all_labels, all_preds, model_arch, model_source, output_dir)

def run_test_evaluation():
    print("Evaluating pre-trained models on the test set...")
    
    data_info = load_data()
    data_root = data_info["data_root"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_configs = [
        ("simple_cnn", "simple_cnn"),
        ("resnet50", "imagenet"),
    ]

    print("\n==============================================")
    print("      Test Set Evaluation")
    print("==============================================")
    for arch, source in model_configs:
        evaluate_on_test_set(arch, source, data_root, device)

if __name__ == "__main__":
    run_test_evaluation()

