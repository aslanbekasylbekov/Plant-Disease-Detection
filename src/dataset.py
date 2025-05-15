import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from .config import DATA_DIR, IMG_SIZE, BATCH_SIZE

def get_dataloaders():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) 
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) 
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), transform=data_transforms[x])
        for x in ['train', 'val']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        for x in ['train', 'val']
    }

    class_names = image_datasets['train'].classes
    return dataloaders, class_names
    

