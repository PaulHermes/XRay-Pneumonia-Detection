import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import learning_curve
from sklearn.metrics import make_scorer, recall_score
from skorch import NeuralNetClassifier

from src import hyperparameters as hp
from src.model import build_model
from src.data_management import PneumoniaDataset, get_train_transforms

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, recall_score
from skorch import NeuralNetClassifier
from collections import defaultdict

from src import hyperparameters as hp
from src.model import build_model
from src.data_management import PneumoniaDataset, get_train_transforms

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
        train_split=None,
        verbose=0,
        device=device,
    )

    # 2. Load data
    train_transforms = get_train_transforms()
    train_dataset = PneumoniaDataset(data_root, transform=train_transforms, split='train')
    X = np.array([item[0].numpy() for item in train_dataset])
    y = np.array([item[1] for item in train_dataset])
    
    # 3. Define training sizes and results storage
    train_sizes_abs = np.linspace(int(0.1 * len(X)), len(X), 5, dtype=int)
    results = defaultdict(lambda: defaultdict(list))
    
    # 4. Iterate over training sizes and compute all metrics at once
    for i, n_samples in enumerate(train_sizes_abs):
        print(f"  - [{i+1}/{len(train_sizes_abs)}] Training on {n_samples} samples...")
        
        X_subset, y_subset = X[:n_samples], y[:n_samples]

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
    train_sizes_proportions = train_sizes_abs[:actual_train_sizes] / len(X)

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

