import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "tomato_model.pth")

BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 1e-4
IMG_SIZE = 224
