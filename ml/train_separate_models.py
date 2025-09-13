#!/usr/bin/env python3
"""
Обучение отдельных моделей для чистоты и повреждений
Структура данных:
data/
├── clean/      # Чистые машины
├── dirty/      # Грязные машины
├── damaged/    # Битые машины
└── intact/     # Целые машины
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import ssl

# Исправление SSL проблемы на macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

class CarDataset(Dataset):
    """Датасет для изображений автомобилей"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Ошибка загрузки изображения {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CarConditionModel(nn.Module):
    """Модель для бинарной классификации"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(CarConditionModel, self).__init__()
        
        # Используем ResNet50 как backbone (исправлено для новых версий)
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Замораживаем ранние слои для transfer learning
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False
        
        # Заменяем классификатор
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def load_cleanliness_data(data_dir):
    """Загрузка данных для модели чистоты"""
    
    image_paths = []
    labels = []
    class_counts = defaultdict(int)
    
    print("Загрузка данных для модели чистоты...")
    
    # Clean = 0, Dirty = 1
    folders = {
        'clean': 0,
        'dirty': 1
    }
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for folder_name, class_id in folders.items():
        folder_path = os.path.join('../data', folder_name)  # Исправлен путь
        
        if not os.path.exists(folder_path):
            print(f"⚠️  Папка {folder_path} не найдена!")
            continue
        
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                image_path = os.path.join(folder_path, filename)
                image_paths.append(image_path)
                labels.append(class_id)
                class_counts[folder_name] += 1
    
    print("\n📊 Статистика данных (чистота):")
    for folder_name, count in class_counts.items():
        print(f"  {folder_name}: {count} изображений")
    print(f"  Всего: {len(image_paths)} изображений")
    
    return image_paths, labels

def load_condition_data(data_dir):
    """Загрузка данных для модели повреждений"""
    
    image_paths = []
    labels = []
    class_counts = defaultdict(int)
    
    print("Загрузка данных для модели повреждений...")
    
    # Intact = 0, Damaged = 1
    folders = {
        'intact': 0,
        'damaged': 1
    }
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for folder_name, class_id in folders.items():
        folder_path = os.path.join('../data', folder_name)  # Исправлен путь
        
        if not os.path.exists(folder_path):
            print(f"⚠️  Папка {folder_path} не найдена!")
            continue
        
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                image_path = os.path.join(folder_path, filename)
                image_paths.append(image_path)
                labels.append(class_id)
                class_counts[folder_name] += 1
    
    print("\n📊 Статистика данных (повреждения):")
    for folder_name, count in class_counts.items():
        print(f"  {folder_name}: {count} изображений")
    print(f"  Всего: {len(image_paths)} изображений")
    
    return image_paths, labels

def get_transforms():
    """Получение трансформаций для обучения и валидации"""
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_model(model, train_loader, val_loader, model_name, num_epochs=30, device='cuda'):
    """Обучение модели"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    best_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    print(f"\n🚀 Начинаем обучение модели {model_name}")
    print(f"Эпох: {num_epochs}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        print(f'\nЭпоха {epoch+1}/{num_epochs}')
        
        # Фаза обучения
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        train_bar = tqdm(train_loader, desc="Обучение")
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            current_acc = running_corrects.double() / total_samples
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        print(f'Обучение - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Фаза валидации
        model.eval()
        val_corrects = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Валидация")
            for inputs, labels in val_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                val_corrects += torch.sum(preds == labels.data)
                val_total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = val_corrects.double() / val_total
        val_accuracies.append(val_acc)
        
        print(f'Валидация - Acc: {val_acc:.4f}')
        
        # Сохранение лучшей модели
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'models/{model_name}_model.pth')
            print(f'✅ Новая лучшая модель сохранена! Точность: {best_acc:.4f}')
        
        scheduler.step(val_acc)
    
    print(f"\n🎉 Обучение {model_name} завершено!")
    print(f"Лучшая точность: {best_acc:.4f}")
    
    return model, best_acc

def main():
    """Основная функция обучения"""
    
    # Настройки
    DATA_DIR = 'unused'  # Не используется, пути исправлены в функциях
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    
    # Создаем папку для моделей
    os.makedirs('models', exist_ok=True)
    
    # Проверка доступности CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Используется устройство: {device}")
    
    # Получение трансформаций
    train_transform, val_transform = get_transforms()
    
    # === ОБУЧЕНИЕ МОДЕЛИ ЧИСТОТЫ ===
    print("\n" + "="*60)
    print("🧽 ОБУЧЕНИЕ МОДЕЛИ ЧИСТОТЫ")
    print("="*60)
    
    try:
        clean_paths, clean_labels = load_cleanliness_data(DATA_DIR)
        
        if len(clean_paths) > 0:
            # Разделение данных
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                clean_paths, clean_labels, test_size=0.2, random_state=42, stratify=clean_labels
            )
            
            # Создание датасетов
            train_dataset = CarDataset(train_paths, train_labels, train_transform)
            val_dataset = CarDataset(val_paths, val_labels, val_transform)
            
            # Создание загрузчиков
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            
            # Создание и обучение модели
            cleanliness_model = CarConditionModel(num_classes=2).to(device)
            cleanliness_model, clean_acc = train_model(
                cleanliness_model, train_loader, val_loader, 
                'cleanliness', NUM_EPOCHS, device
            )
        else:
            print("❌ Не найдено данных для модели чистоты!")
            clean_acc = 0
    
    except Exception as e:
        print(f"❌ Ошибка при обучении модели чистоты: {e}")
        clean_acc = 0
    
    # === ОБУЧЕНИЕ МОДЕЛИ ПОВРЕЖДЕНИЙ ===
    print("\n" + "="*60)
    print("🔧 ОБУЧЕНИЕ МОДЕЛИ ПОВРЕЖДЕНИЙ")
    print("="*60)
    
    try:
        condition_paths, condition_labels = load_condition_data(DATA_DIR)
        
        if len(condition_paths) > 0:
            # Разделение данных
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                condition_paths, condition_labels, test_size=0.2, random_state=42, stratify=condition_labels
            )
            
            # Создание датасетов
            train_dataset = CarDataset(train_paths, train_labels, train_transform)
            val_dataset = CarDataset(val_paths, val_labels, val_transform)
            
            # Создание загрузчиков
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            
            # Создание и обучение модели
            condition_model = CarConditionModel(num_classes=2).to(device)
            condition_model, condition_acc = train_model(
                condition_model, train_loader, val_loader, 
                'condition', NUM_EPOCHS, device
            )
        else:
            print("❌ Не найдено данных для модели повреждений!")
            condition_acc = 0
    
    except Exception as e:
        print(f"❌ Ошибка при обучении модели повреждений: {e}")
        condition_acc = 0
    
    # === ИТОГИ ===
    print("\n" + "="*60)
    print("🎉 ИТОГИ ОБУЧЕНИЯ")
    print("="*60)
    print(f"🧽 Модель чистоты: {clean_acc:.4f}")
    print(f"🔧 Модель повреждений: {condition_acc:.4f}")
    print(f"📁 Модели сохранены в папке: models/")
    print(f"🔮 Для тестирования используйте: python analyze.py <путь_к_изображению>")

if __name__ == "__main__":
    main()