import warnings
warnings.filterwarnings('ignore') 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn.functional as F
import colorama
from colorama import Fore, Style


Root_dir = "/Users/aslanbekasylbekov/Desktop/diploma/tomato-disease-classifier/data"
train_dir = Root_dir + "/train"
valid_dir = Root_dir + "/val"
test_dir = "/Users/aslanbekasylbekov/Desktop/diploma/tomato-disease-classifier/test"
Diseases_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith(".")]
# print(Fore.GREEN +str(Diseases_classes))
# print("\nTotal number of classes are: ", len(Diseases_classes))

# plt.figure(figsize=(20, 20))  # Уменьшил размер, чтобы было удобнее
# cnt = 0
# plant_names = []
# tot_images = 0

# rows, cols = 7, 7  # Количество строк и столбцов в сетке (изменяй под свои данные)
# num_classes = len(Diseases_classes)

# for i in Diseases_classes:
#     if cnt >= rows * cols:
#         break  # Останавливаемся, если достигли лимита ячеек

#     cnt += 1
#     plant_names.append(i)

#     image_path = os.listdir(os.path.join(train_dir, i))
#     print(Fore.GREEN + f"The Number of Images in {i}: {len(image_path)}", end=" ")
#     tot_images += len(image_path)

#     img_show = plt.imread(os.path.join(train_dir, i, image_path[0]))

#     plt.subplot(rows, cols, cnt)
#     plt.imshow(img_show)
#     plt.title(i, fontsize=12)
#     plt.axis("off")  # Убираем оси

# plt.tight_layout()  # Автоупорядочивание для предотвращения наложений
# plt.show()  # Показываем результат
    
    
# print("\nTotal Number of Images in Directory: ", tot_images)

# plant_names = []
# Len = []

# for i in Diseases_classes:
#     plant_names.append(i)
#     imgs_path = os.listdir(os.path.join(train_dir, i))
#     Len.append(len(imgs_path))

# # Сортируем данные по убыванию количества изображений
# sorted_data = sorted(zip(Len, plant_names), reverse=True)
# Len, plant_names = zip(*sorted_data)

# # Настройка графика
# sns.set(style="whitegrid", color_codes=True)
# plt.figure(figsize=(12, 8), dpi=100)  # Уменьшаем размер графика
# ax = sns.barplot(x=Len, y=plant_names, palette="Greens")

# # Настройка подписей
# plt.xlabel("Количество изображений", fontsize=14)
# plt.ylabel("Классы заболеваний", fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# plt.show()

train = ImageFolder(train_dir, transform=transforms.ToTensor())
valid = ImageFolder(valid_dir, transform=transforms.ToTensor())
# # print(train[0])
# def show_image(image, label):
#     class_name = train.classes[label]  # Получаем название класса
#     print(f"Label: {class_name} ({label})")
#     plt.imshow(image.permute(1, 2, 0))  # Меняем порядок каналов
#     plt.title(class_name, fontsize=10)  # Используем имя класса в заголовке

# # Список индексов изображений
# image_list = [0, 3000, 5000, 8000, 12000, 15000, 60000, 70000]

# # Создаем фигуру
# plt.figure(figsize=(12, 6))  
# chs = 0

# for img_idx in image_list:
#     if img_idx >= len(train):  # Проверяем, существует ли индекс
#         print(Fore.RED + f"Warning: Index {img_idx} is out of range!")
#         continue
    
#     chs += 1
#     plt.subplot(2, 4, chs)
    
#     img, label = train[img_idx]  # Получаем изображение и метку
#     show_image(img, label)

# plt.tight_layout()  # Вызываем после всех графиков
# plt.show()

batch_size = 32
# Функция для выбора устройства (GPU или CPU)
def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Функция для переноса данных или модели на устройство
def to_device(data, device):
    if isinstance(data, nn.Module):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# Функция-обертка для DataLoader, чтобы перемещать данные на GPU/CPU
class DeviceDataLoader():
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        
    def __iter__(self):
        for b in self.dataloader:
            yield to_device(b, self.device)
        
    def __len__(self):
        return len(self.dataloader)

# Загружаем данные
train_dir = Root_dir + "/train"
valid_dir = Root_dir + "/val"

train = ImageFolder(train_dir, transform=transforms.ToTensor())
valid = ImageFolder(valid_dir, transform=transforms.ToTensor())

# Проверяем классы
print("Classes in train dataset:", train.classes)

# Настройки
batch_size = 32
device = get_default_device()
use_pin_memory = device.type == "cuda"  # pin_memory только для GPU

# Создаем DataLoader
train_dataloader = DataLoader(train, batch_size, shuffle=True, num_workers=2, pin_memory=use_pin_memory)
valid_dataloader = DataLoader(valid, batch_size, num_workers=2, pin_memory=use_pin_memory)

# Оборачиваем DataLoader для автоматического переноса данных
train_dataloader = DeviceDataLoader(train_dataloader, device)
valid_dataloader = DeviceDataLoader(valid_dataloader, device)

# Определяем модель
class CNN_NeuralNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(4))
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(128 * 7 * 7, num_classes))  # Размер зависит от входных данных

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

# Создаем модель и переносим на устройство
model = CNN_NeuralNet(3, len(train.classes))
model = to_device(model, device)

# print(model)
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def fit_OneCycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0,
                grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []  #For collecting the results
    
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # scheduler for one cycle learniing rate
    #Sets the learning rate of each parameter group according to the 1cycle learning rate policy. 
    #The 1cycle policy anneals the learning rate from an initial learning rate to some 
    #maximum learning rate and then from that maximum learning rate to some minimum learning rate
    #much lower than the initial learning rate. 
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr,
                                                epochs=epochs, steps_per_epoch=len(train_loader))
    
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # gradient clipping
            #Clip the gradients of an iterable of parameters at specified value.
            #All from pytorch documantation.
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            
            # recording and updating learning rates
            lrs.append(get_lr(optimizer))
            sched.step()
             # validation
        
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
        
    return history

history = [evaluate(model, valid_dataloader)]
print(history)