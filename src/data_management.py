import kagglehub
import os
from pathlib import Path

def load_data():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "raw"

    os.environ['KAGGLEHUB_CACHE'] = str(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    
    print(f"Dataset download complete. Path to dataset files: {path}")

    return path

