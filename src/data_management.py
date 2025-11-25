import kagglehub
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import hashlib
from collections import defaultdict
import random
import shutil

def load_data():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "raw"

    os.environ['KAGGLEHUB_CACHE'] = str(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading 'chest-xray-pneumonia' dataset from Kaggle...")
    version_path = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia"))

    # Clean up junk __MACOSX directories
    toplevel_macosx_path = version_path / '__MACOSX'
    if toplevel_macosx_path.is_dir():
        print(f"Found and deleting junk directory: {toplevel_macosx_path}")
        shutil.rmtree(toplevel_macosx_path)

    # The unzipped data is often inside a folder named after the dataset
    data_root = version_path / 'chest_xray'
    if not data_root.is_dir():
        data_root = version_path # Fallback
    
    nested_macosx_path = data_root / '__MACOSX'
    if nested_macosx_path.is_dir():
        print(f"Found and deleting junk directory: {nested_macosx_path}")
        shutil.rmtree(nested_macosx_path)
        
    nested_dir = data_root / 'chest_xray'
    
    if (data_root / 'train').is_dir() and nested_dir.is_dir():
        print(f"Found redundant nested directory: {nested_dir}. Deleting it.")
        shutil.rmtree(nested_dir)
    elif not (data_root / 'train').is_dir() and nested_dir.is_dir():
        print(f"Data found only in nested directory. Using {nested_dir} as root.")
        data_root = nested_dir

    print(f"Dataset ready. Root folder with train/test/val is: {data_root}")
    return data_root

def get_data_summary(data_path: Path) -> pd.DataFrame:
    data_path = Path(data_path)
    summary = defaultdict(lambda: defaultdict(int))
    for split in ['train', 'val', 'test']:
        for category in ['NORMAL', 'PNEUMONIA']:
            dir_path = data_path / split / category
            if dir_path.is_dir():
                count = len([f for f in dir_path.iterdir() if f.is_file()])
                summary[split][category] = count
    
    df = pd.DataFrame.from_dict(summary, orient='index')
    df['TOTAL'] = df.sum(axis=1)
    df.loc['TOTAL'] = df.sum()
    return df

def verify_images(data_path: Path, num_samples_to_check: int = 100):
    data_path = Path(data_path)
    results = {
        "corrupted_files": [],
        "dimensions": set(),
        "modes": defaultdict(int),
        "paths_by_mode": defaultdict(list),
        "total_checked": 0
    }
    image_paths = list(data_path.glob('**/*.jpeg'))
    
    if not image_paths:
        return results, []

    sample_size = min(len(image_paths), num_samples_to_check)
    results['total_checked'] = sample_size
    sample_paths = random.sample(image_paths, sample_size)

    for img_path in sample_paths:
        try:
            with Image.open(img_path) as img:
                mode = img.mode
                results["dimensions"].add(img.size)
                results["modes"][mode] += 1
                results["paths_by_mode"][mode].append(img_path)
        except Exception:
            results["corrupted_files"].append(str(img_path))
    
    return results, sample_paths

def analyze_pneumonia_subtypes(data_path: Path) -> tuple[pd.DataFrame, list]:
    data_path = Path(data_path)
    subtypes = defaultdict(lambda: defaultdict(int))
    unknown_paths = []
    for split in ['train', 'val', 'test']:
        pneumonia_dir = data_path / split / 'PNEUMONIA'
        if pneumonia_dir.is_dir():
            for img_path in pneumonia_dir.iterdir():
                if not img_path.is_file(): continue
                if img_path.name == '.DS_Store': continue # Exclude macOS .DS_Store files
                
                filename = img_path.name.lower()
                if 'virus' in filename:
                    subtypes[split]['virus'] += 1
                elif 'bacteria' in filename:
                    subtypes[split]['bacteria'] += 1
                else:
                    subtypes[split]['unknown'] += 1
                    unknown_paths.append(img_path)
    
    df = pd.DataFrame.from_dict(subtypes, orient='index').fillna(0).astype(int)
    df['TOTAL'] = df.sum(axis=1)
    df.loc['TOTAL'] = df.sum()
    return df, unknown_paths

def _hash_file(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(8192)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(8192)
    return hasher.hexdigest()

def check_for_data_leakage(data_path: Path) -> dict:
    data_path = Path(data_path)
    hashes = defaultdict(set)
    leaks = {}

    for split in ['train', 'val', 'test']:
        split_path = data_path / split
        if not split_path.is_dir(): continue
        
        image_paths = list(split_path.glob('**/*.jpeg'))
        for img_path in image_paths:
            hashes[split].add(_hash_file(img_path))
    
    leaks["train_val"] = len(hashes['train'].intersection(hashes['val']))
    leaks["train_test"] = len(hashes['train'].intersection(hashes['test']))
    leaks["val_test"] = len(hashes['val'].intersection(hashes['test']))
    
    return leaks

