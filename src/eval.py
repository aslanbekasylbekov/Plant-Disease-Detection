import torch
from .model import build_model
from .dataset import get_dataloaders
from .config import MODEL_PATH
from .utils import evaluate_model

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders, class_names = get_dataloaders()
    model = build_model(len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    evaluate_model(model, dataloaders['val'], device, class_names)
