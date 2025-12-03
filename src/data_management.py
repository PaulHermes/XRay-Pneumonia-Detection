import kagglehub
import os
from pathlib import Path
from PIL import Image
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

def preprocess_image(image_path: Path, target_size: tuple[int, int]) -> Image.Image:
    img = Image.open(image_path).convert('RGB')
    
    original_width, original_height = img.size
    
    if original_width == original_height:
        return img.resize(target_size, Image.Resampling.LANCZOS)

    max_side = max(original_width, original_height)
    padded_img = Image.new('RGB', (max_side, max_side), (0, 0, 0))
    
    paste_x = (max_side - original_width) // 2
    paste_y = (max_side - original_height) // 2
    padded_img.paste(img, (paste_x, paste_y))
    
    resized_img = padded_img.resize(target_size, Image.Resampling.LANCZOS)
    
    return resized_img