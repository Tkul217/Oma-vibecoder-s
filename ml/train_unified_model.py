#!/usr/bin/env python3
"""
Обучение единой модели для анализа состояния автомобилей
Модель определяет 4 класса одновременно:
- clean_intact (чистый целый)
- clean_damaged (чистый битый)  
- dirty_intact (грязный целый)
- dirty_damaged (грязный битый)

Структура данных:
data/
├── clean_intact/     # Чистые целые машины
├── clean_damaged/    # Чистые битые машины
├── dirty_intact/     # Грязные целые машины
└── dirty_damaged/    # Грязные битые машины
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
import shutil

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
            # Создаем пустое изображение в случае ошибки
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class UnifiedCarModel(nn.Module):
    """Единая модель для классификации состояния автомобилей (4 класса)"""
    
    def __init__(self, num_classes=4, pretrained=True):
        super(UnifiedCarModel, self).__init__()
        
        # Используем ResNet50 как backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
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

def load_data_from_folders(data_dir):
    """Загрузка данных из 4 папок"""
    
    # Маппинг папок на классы
    class_mapping = {
        'clean_intact': 0,     # чистый целый
        'clean_damaged': 1,    # чистый битый
        'dirty_intact': 2,     # грязный целый
        'dirty_damaged': 3     # грязный битый
    }
    
    image_paths = []
    labels = []
    class_counts = defaultdict(int)
    
    print("Загрузка данных из папок...")
    
    for folder_name, class_id in class_mapping.items():
        folder_path = os.path.join('../data', folder_name)

        if not os.path.exists(folder_path):
            print(f"⚠️  Папка {folder_path} не найдена!")
            continue

        # Поддерживаемые форматы изображений
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                image_path = os.path.join(folder_path, filename)
                image_paths.append(image_path)
                labels.append(class_id)
                class_counts[folder_name] += 1

    print("\n📊 Статистика данных:")
    for folder_name, count in class_counts.items():
        print(f"  {folder_name}: {count} изображений")
    print(f"  Всего: {len(image_paths)} изображений")

    return image_paths, labels, class_mapping

def get_transforms():
    """Получение трансформаций для обучения и валидации"""

    # Аугментации для обучения
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

    # Трансформации для валидации
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda', save_path='models/'):
    """Обучение модели"""

    # Создаем папку для сохранения моделей
    os.makedirs(save_path, exist_ok=True)

    # Настройка оптимизатора и функции потерь
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    best_acc = 0.0
    train_losses = []
    val_accuracies = []

    print(f"\n🚀 Начинаем обучение на {device}")
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

            # Обновляем прогресс-бар
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
            torch.save(model.state_dict(), os.path.join(save_path, 'best_unified_model.pth'))
            print(f'✅ Новая лучшая модель сохранена! Точность: {best_acc:.4f}')

        # Обновление learning rate
        scheduler.step(val_acc)

        # Сохранение промежуточной модели каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch+1}.pth'))

    print(f"\n🎉 Обучение завершено!")
    print(f"Лучшая точность: {best_acc:.4f}")

    # Построение графиков
    plot_training_history(train_losses, val_accuracies, save_path)

    # Отчет по классификации
    class_names = ['clean_intact', 'clean_damaged', 'dirty_intact', 'dirty_damaged']
    print("\n📊 Отчет по классификации:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Матрица ошибок
    plot_confusion_matrix(all_labels, all_preds, class_names, save_path)

    return model

def plot_training_history(train_losses, val_accuracies, save_path):
    """Построение графиков обучения"""

    plt.figure(figsize=(15, 5))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot([acc.cpu().numpy() for acc in val_accuracies], 'r-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("📈 Графики обучения сохранены в training_history.png")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Построение матрицы ошибок"""

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("📊 Матрица ошибок сохранена в confusion_matrix.png")

def create_inference_script(class_mapping, save_path):
    """Создание скрипта для инференса"""

    inference_code = f'''#!/usr/bin/env python3
"""
Скрипт для инференса обученной модели
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import sys
import os

# Маппинг классов
CLASS_MAPPING = {class_mapping}
REVERSE_MAPPING = {{v: k for k, v in CLASS_MAPPING.items()}}

class UnifiedCarModel(torch.nn.Module):
    def __init__(self, num_classes=4):
        super(UnifiedCarModel, self).__init__()
        from torchvision import models

        self.backbone = models.resnet50(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_features, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def load_model(model_path):
    model = UnifiedCarModel(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        class_name = REVERSE_MAPPING[predicted.item()]

        # Разбираем класс на компоненты
        parts = class_name.split('_')
        cleanliness = parts[0]  # clean или dirty
        condition = parts[1]    # intact или damaged

        return {{
            'cleanliness': cleanliness,
            'cleanlinessConfidence': confidence.item(),
            'condition': condition,
            'conditionConfidence': confidence.item(),
            'full_class': class_name,
            'class_probabilities': {{
                REVERSE_MAPPING[i]: prob.item()
                for i, prob in enumerate(probabilities[0])
            }}
        }}

def main():
    if len(sys.argv) != 2:
        print(json.dumps({{"error": "Usage: python inference.py <image_path>"}}))
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = "best_unified_model.pth"

    try:
        model = load_model(model_path)
        image_tensor = preprocess_image(image_path)
        result = predict(model, image_tensor)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({{"error": str(e)}}))
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

    with open(os.path.join(save_path, 'inference.py'), 'w', encoding='utf-8') as f:
        f.write(inference_code)

    print("🔮 Скрипт для инференса создан: inference.py")

def main():
    """Основная функция обучения"""

    # Настройки
    DATA_DIR = '../data'  # Папка с данными
    MODELS_DIR = 'models'
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    
    # Проверка доступности CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Используется устройство: {device}")
    
    # Загрузка данных
    try:
        image_paths, labels, class_mapping = load_data_from_folders(DATA_DIR)
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        print("\n📁 Убедитесь, что структура папок следующая:")
        print("data/")
        print("├── clean_intact/")
        print("├── clean_damaged/")
        print("├── dirty_intact/")
        print("└── dirty_damaged/")
        return
    
    if len(image_paths) == 0:
        print("❌ Не найдено изображений для обучения!")
        return
    
    # Разделение на обучение и валидацию
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\n📊 Разделение данных:")
    print(f"  Обучение: {len(train_paths)} изображений")
    print(f"  Валидация: {len(val_paths)} изображений")
    
    # Получение трансформаций
    train_transform, val_transform = get_transforms()
    
    # Создание датасетов
    train_dataset = CarDataset(train_paths, train_labels, train_transform)
    val_dataset = CarDataset(val_paths, val_labels, val_transform)
    
    # Создание загрузчиков данных
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Создание модели
    model = UnifiedCarModel(num_classes=4, pretrained=True).to(device)
    
    print(f"\n🧠 Архитектура модели:")
    print(f"  Backbone: ResNet50")
    print(f"  Классы: {len(class_mapping)}")
    print(f"  Параметры: {sum(p.numel() for p in model.parameters()):,}")
    
    # Обучение модели
    trained_model = train_model(model, train_loader, val_loader, 
                               num_epochs=NUM_EPOCHS, device=device, 
                               save_path=MODELS_DIR)
    
    # Создание скрипта для инференса
    create_inference_script(class_mapping, MODELS_DIR)
    
    print(f"\n✅ Обучение завершено!")
    print(f"📁 Модели сохранены в папке: {MODELS_DIR}")
    print(f"🔮 Для тестирования используйте: python {MODELS_DIR}/inference.py <путь_к_изображению>")

if __name__ == "__main__":
    main()