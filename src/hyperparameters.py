# Model Hyperparameters
MODEL_ARCH = "simple_cnn" # "simple_cnn" or "resnet50"
MODEL_SOURCE = "simple_cnn" # "imagenet" or "hf_pretrained" or "simple_cnn"
HF_MODEL_PATH = "ryefoxlime/PneumoniaDetection"
PRETRAINED = True # Use pretrained weights from source
NUM_CLASSES = 2

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
optimizer_params = {
    "weight_decay": 0.0001
}
scheduler_params = {
    "mode": "min",
    "factor": 0.1,
    "patience": 5
}

# Data Hyperparameters
TARGET_IMAGE_SIZE = (224, 224)
TRANSFORM_CONFIGS = {
    "imagenet": {
        "MEAN": [0.485, 0.456, 0.406],
        "STD": [0.229, 0.224, 0.225],
    },
    "hf_pretrained": {
        "MEAN": [0.5, 0.5, 0.5],
        "STD": [0.5, 0.5, 0.5],
    },
    "simple_cnn": {
        "MEAN": [0.5, 0.5, 0.5],
        "STD": [0.5, 0.5, 0.5],
    }
}

#zuerst ohne (impact schauen)
# Data Augmentation Hyperparameters
AUGMENTATION_CONFIG = {
    "enabled": False,
    "rotation": {"enabled": True, "degrees": 5},
    "affine": {"enabled": True, "translate": (0.05, 0.05), "scale": (0.95, 1.05)},
    "horizontal_flip": {"enabled": True, "p": 0.5},
    "color_jitter": {"enabled": True, "brightness": 0.1, "contrast": 0.1},
}


# R rauskriegen (aufheben für später)
 
# model weights, gdrive