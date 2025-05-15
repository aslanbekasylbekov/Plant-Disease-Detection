import torch
import os
from tqdm import tqdm
from .dataset import get_dataloaders
from .model import build_model
from .config import EPOCHS, LEARNING_RATE, MODEL_PATH, MODEL_DIR
from .utils import save_model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders, class_names = get_dataloaders()
    model = build_model(len(class_names)).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(dataloaders['train']):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloaders['train'])
        print(f"Train Loss: {epoch_loss:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)