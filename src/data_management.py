import kagglehub
import os
import random
from pathlib import Path
from PIL import Image
import shutil
import torchvision.transforms as transforms
import torch
from . import hyperparameters as hp
from collections import defaultdict

def resplit_data(data_root: Path, val_split_ratio: float = 0.2):
    print("Checking data split...")
    train_dir = data_root / 'train'
    val_dir = data_root / 'val'
    
    if val_dir.is_dir():
        try:
            val_normal_count = len(list((val_dir / 'NORMAL').iterdir()))
            if val_normal_count > 100: # Original val set has only 8
                print("Validation set appears to be already split. Skipping resplit.")
                return
        except FileNotFoundError:
            pass 
    val_normal_dir = val_dir / 'NORMAL'
    val_pneumonia_dir = val_dir / 'PNEUMONIA'
    val_normal_dir.mkdir(parents=True, exist_ok=True)
    val_pneumonia_dir.mkdir(parents=True, exist_ok=True)

    for class_name in ['NORMAL', 'PNEUMONIA']:
        orig_val_class_dir = val_dir / class_name
        train_class_dir = train_dir / class_name
        if orig_val_class_dir.is_dir():
            for img_path in orig_val_class_dir.iterdir():
                if img_path.is_file():
                    shutil.move(str(img_path), train_class_dir / img_path.name)

    for class_name in ['NORMAL', 'PNEUMONIA']:
        source_dir = train_dir / class_name
        all_images = list(source_dir.glob('*.jpeg'))
        random.shuffle(all_images)
        
        num_val_images = int(len(all_images) * val_split_ratio)
        val_images = all_images[:num_val_images]
        
        destination_dir = val_dir / class_name
        for img_path in val_images:
            shutil.move(str(img_path), destination_dir / img_path.name)
            
        print(f"Moved {len(val_images)} images from train/{class_name} to val/{class_name}")

    print("Data resplit complete.")


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
    
    # Perform the data resplit
    resplit_data(data_root)

    # get class counts for the training split for weighted sampling
    train_normal_count = len(list((data_root / 'train' / 'NORMAL').iterdir()))
    train_pneumonia_count = len(list((data_root / 'train' / 'PNEUMONIA').iterdir()))
    
    class_counts = {
        'NORMAL': train_normal_count,
        'PNEUMONIA': train_pneumonia_count
    }
    
    return {'data_root': data_root, 'class_counts': class_counts}

class ResizeAndPad: #class to be callable from compose
    def __init__(self, target_size: tuple[int, int]):
        self.target_size = target_size

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert('RGB')
        original_width, original_height = img.size
        
        if original_width == original_height:
            return img.resize(self.target_size, Image.Resampling.LANCZOS)

        max_side = max(original_width, original_height)
        padded_img = Image.new('RGB', (max_side, max_side), (0, 0, 0))
        
        paste_x = (max_side - original_width) // 2
        paste_y = (max_side - original_height) // 2
        padded_img.paste(img, (paste_x, paste_y))
        
        resized_img = padded_img.resize(self.target_size, Image.Resampling.LANCZOS)
        
        return resized_img

def get_train_transforms() -> transforms.Compose:
    transform_params = hp.TRANSFORM_CONFIGS.get(hp.MODEL_SOURCE, hp.TRANSFORM_CONFIGS["imagenet"])
    mean = transform_params["MEAN"]
    std = transform_params["STD"]
    
    # build augmentation list
    aug_list = []
    if hp.AUGMENTATION_CONFIG.get("enabled", False):
        cfg = hp.AUGMENTATION_CONFIG
        if cfg.get("rotation", {}).get("enabled", False):
            aug_list.append(transforms.RandomRotation(degrees=cfg["rotation"]["degrees"]))
        if cfg.get("affine", {}).get("enabled", False):
            aug_list.append(transforms.RandomAffine(
                degrees=0, # Rotation is handled separately
                translate=cfg["affine"]["translate"],
                scale=cfg["affine"]["scale"]
            ))
        if cfg.get("horizontal_flip", {}).get("enabled", False):
            aug_list.append(transforms.RandomHorizontalFlip(p=cfg["horizontal_flip"]["p"]))
        if cfg.get("color_jitter", {}).get("enabled", False):
            aug_list.append(transforms.ColorJitter(
                brightness=cfg["color_jitter"]["brightness"],
                contrast=cfg.get("color_jitter", {}).get("contrast", 0) # Contrast is optional
            ))

    # Base transforms
    base_transforms = [
        ResizeAndPad(hp.TARGET_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    
    # Combine augmentations with base transforms
    return transforms.Compose(aug_list + base_transforms)

def get_val_transforms() -> transforms.Compose:
    transform_params = hp.TRANSFORM_CONFIGS.get(hp.MODEL_SOURCE, hp.TRANSFORM_CONFIGS["imagenet"])
    mean = transform_params["MEAN"]
    std = transform_params["STD"]

    return transforms.Compose([
        ResizeAndPad(hp.TARGET_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

class PneumoniaDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.class_to_idx = {'NORMAL': 0, 'PNEUMONIA': 1}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        instances = []
        for class_name in self.classes:
            class_dir = self.root_dir / self.split / class_name
            if not class_dir.is_dir():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() == '.jpeg':
                    label = self.class_to_idx[class_name]
                    instances.append((str(img_path), label))
        return instances

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def get_weighted_sampler(dataset: PneumoniaDataset) -> torch.utils.data.WeightedRandomSampler:
    class_counts = defaultdict(int)
    for _, label in dataset.samples:
        class_counts[label] += 1

    num_samples = len(dataset)
    weights = [0] * num_samples
    
    class_weights = {
        0: num_samples / (2 * class_counts[0]), # Weight for NORMAL
        1: num_samples / (2 * class_counts[1])  # Weight for PNEUMONIA
    }

    for idx, (_, label) in enumerate(dataset.samples):
        weights[idx] = class_weights[label]
            
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.DoubleTensor(weights),
        num_samples=num_samples,
        replacement=True
    )
    return sampler

def create_training_dataset_with_sampler(data_root: Path):
    train_transforms = get_train_transforms()
    train_dataset = PneumoniaDataset(data_root, transform=train_transforms, split='train')
    train_sampler = get_weighted_sampler(train_dataset)
    return train_dataset, train_sampler