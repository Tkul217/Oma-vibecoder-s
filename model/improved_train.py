#!/usr/bin/env python3
"""
Улучшенное обучение модели с продвинутыми техниками
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import pandas as pd
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
import warnings
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from improved_car_model import ImprovedCarConditionClassifier
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torchvision.transforms.functional as TF

warnings.filterwarnings('ignore')

class AdvancedCarDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, is_training=True):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_training = is_training
        
        print(f"📊 Загружено примеров: {len(self.data)}")
        print(f"   Чистые: {len(self.data[self.data['cleanliness'] == 'clean'])}")
        print(f"   Грязные: {len(self.data[self.data['cleanliness'] == 'dirty'])}")
        print(f"   Целые: {len(self.data[self.data['damage'] == 'intact'])}")
        print(f"   Битые: {len(self.data[self.data['damage'] == 'damaged'])}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Загрузка изображения
        img_path = self.root_dir / row['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Ошибка загрузки {img_path}: {e}")
            # Создаем заглушку
            image = Image.new('RGB', (256, 256), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        # Метки (0/1)
        cleanliness_label = 0 if row['cleanliness'] == 'clean' else 1
        damage_label = 0 if row['damage'] == 'intact' else 1
        
        # Комбинированная метка для общего классификатора
        overall_label = cleanliness_label * 2 + damage_label
        
        return {
            'image': image,
            'cleanliness': torch.tensor(cleanliness_label, dtype=torch.long),
            'damage': torch.tensor(damage_label, dtype=torch.long),
            'overall': torch.tensor(overall_label, dtype=torch.long)
        }

class FocalLoss(nn.Module):
    """Focal Loss для работы с несбалансированными данными"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EarlyStopping:
    """Early Stopping для предотвращения переобучения"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def get_advanced_transforms(is_training=True):
    """Продвинутые трансформации для аугментации"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def calculate_class_weights(dataset):
    """Вычисление весов классов для балансировки"""
    cleanliness_labels = [dataset[i]['cleanliness'].item() for i in range(len(dataset))]
    damage_labels = [dataset[i]['damage'].item() for i in range(len(dataset))]
    
    # Веса для чистоты
    clean_count = cleanliness_labels.count(0)
    dirty_count = cleanliness_labels.count(1)
    cleanliness_weights = torch.tensor([dirty_count / clean_count, clean_count / dirty_count], dtype=torch.float)
    
    # Веса для повреждений
    intact_count = damage_labels.count(0)
    damaged_count = damage_labels.count(1)
    damage_weights = torch.tensor([damaged_count / intact_count, intact_count / damaged_count], dtype=torch.float)
    
    return cleanliness_weights, damage_weights

def train_improved_model(epochs=30, batch_size=16, lr=0.001, patience=10):
    """Улучшенное обучение модели"""
    
    print("🚗 Улучшенное обучение модели анализа автомобилей")
    print("=" * 55)
    
    # Проверяем наличие данных
    if not os.path.exists('data/dataset.csv'):
        print("❌ Файл data/dataset.csv не найден!")
        print("💡 Сначала запустите: python prepare_4folders.py")
        return False
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Устройство: {device}")
    
    # Трансформации
    train_transform = get_advanced_transforms(is_training=True)
    val_transform = get_advanced_transforms(is_training=False)
    
    # Загрузка данных
    print("\n📊 Подготовка данных...")
    
    df = pd.read_csv('data/dataset.csv')
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    
    print(f"   Обучающих: {len(train_df)}")
    print(f"   Валидационных: {len(val_df)}")
    
    if len(train_df) < 8:
        print("⚠️ Мало обучающих данных! Рекомендуется минимум 50-100 изображений")
    
    # Временные CSV файлы
    train_df.to_csv('train_temp.csv', index=False)
    val_df.to_csv('val_temp.csv', index=False)
    
    # Датасеты
    train_dataset = AdvancedCarDataset('train_temp.csv', 'data', train_transform, is_training=True)
    val_dataset = AdvancedCarDataset('val_temp.csv', 'data', val_transform, is_training=False)
    
    # Вычисляем веса классов
    cleanliness_weights, damage_weights = calculate_class_weights(train_dataset)
    cleanliness_weights = cleanliness_weights.to(device)
    damage_weights = damage_weights.to(device)
    
    print(f"   Веса чистоты: {cleanliness_weights}")
    print(f"   Веса повреждений: {damage_weights}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Модель
    print("\n🤖 Создание улучшенной модели...")
    model = ImprovedCarConditionClassifier()
    model.to(device)
    
    # Критерии с весами классов
    cleanliness_criterion = FocalLoss(alpha=1, gamma=2)
    damage_criterion = FocalLoss(alpha=1, gamma=2)
    overall_criterion = nn.CrossEntropyLoss()
    
    # Оптимизатор с weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    
    # История
    history = {
        'train_loss': [], 'val_loss': [], 'val_acc': [],
        'clean_acc': [], 'damage_acc': [], 'overall_acc': [],
        'lr': []
    }
    best_val_acc = 0.0
    
    print(f"\n🎯 Начало улучшенного обучения на {epochs} эпох...")
    print("=" * 55)
    
    for epoch in range(epochs):
        # ОБУЧЕНИЕ
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        print(f"\nЭпоха {epoch+1}/{epochs}")
        pbar = tqdm(train_loader, desc="Обучение")
        
        for batch in pbar:
            images = batch['image'].to(device)
            clean_labels = batch['cleanliness'].to(device)
            damage_labels = batch['damage'].to(device)
            overall_labels = batch['overall'].to(device)
            
            optimizer.zero_grad()
            
            # Прямой проход
            clean_logits, damage_logits, overall_logits = model(images)
            
            # Loss с весами
            clean_loss = cleanliness_criterion(clean_logits, clean_labels)
            damage_loss = damage_criterion(damage_logits, damage_labels)
            overall_loss = overall_criterion(overall_logits, overall_labels)
            
            # Комбинированный loss
            total_loss = clean_loss + damage_loss + 0.5 * overall_loss
            
            # Обратный проход
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += total_loss.item()
            train_batches += 1
            
            pbar.set_postfix({'loss': total_loss.item(), 'lr': optimizer.param_groups[0]['lr']})
        
        # ВАЛИДАЦИЯ
        model.eval()
        val_loss = 0.0
        correct_clean = 0
        correct_damage = 0
        correct_overall = 0
        total = 0
        
        all_clean_preds = []
        all_clean_labels = []
        all_damage_preds = []
        all_damage_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Валидация")
            for batch in pbar:
                images = batch['image'].to(device)
                clean_labels = batch['cleanliness'].to(device)
                damage_labels = batch['damage'].to(device)
                overall_labels = batch['overall'].to(device)
                
                clean_logits, damage_logits, overall_logits = model(images)
                
                # Loss
                clean_loss = cleanliness_criterion(clean_logits, clean_labels)
                damage_loss = damage_criterion(damage_logits, damage_labels)
                overall_loss = overall_criterion(overall_logits, overall_labels)
                total_loss = clean_loss + damage_loss + 0.5 * overall_loss
                val_loss += total_loss.item()
                
                # Предсказания
                _, clean_pred = torch.max(clean_logits, 1)
                _, damage_pred = torch.max(damage_logits, 1)
                _, overall_pred = torch.max(overall_logits, 1)
                
                # Точность
                correct_clean += (clean_pred == clean_labels).sum().item()
                correct_damage += (damage_pred == damage_labels).sum().item()
                correct_overall += (overall_pred == overall_labels).sum().item()
                total += images.size(0)
                
                # Сохраняем для метрик
                all_clean_preds.extend(clean_pred.cpu().numpy())
                all_clean_labels.extend(clean_labels.cpu().numpy())
                all_damage_preds.extend(damage_pred.cpu().numpy())
                all_damage_labels.extend(damage_labels.cpu().numpy())
        
        # Метрики
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / len(val_loader)
        clean_acc = correct_clean / total
        damage_acc = correct_damage / total
        overall_acc = correct_overall / total
        combined_acc = (clean_acc + damage_acc + overall_acc) / 3
        
        # Обновляем scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            print(f"📉 Learning rate изменен: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Сохранение истории
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(combined_acc)
        history['clean_acc'].append(clean_acc)
        history['damage_acc'].append(damage_acc)
        history['overall_acc'].append(overall_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Вывод результатов
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Clean Acc: {clean_acc:.4f} | Damage Acc: {damage_acc:.4f} | Overall Acc: {overall_acc:.4f}")
        print(f"Combined Acc: {combined_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Сохранение лучшей модели
        if combined_acc > best_val_acc:
            best_val_acc = combined_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': combined_acc,
                'clean_acc': clean_acc,
                'damage_acc': damage_acc,
                'overall_acc': overall_acc,
                'history': history
            }, 'best_improved_model.pth')
            print(f"✅ Сохранена лучшая модель! Точность: {combined_acc:.4f}")
        
        # Early stopping
        if early_stopping(avg_val_loss, model):
            print(f"🛑 Early stopping на эпохе {epoch+1}")
            break
    
    # Детальные метрики
    print("\n📊 Детальные метрики:")
    print("Чистота:")
    print(classification_report(all_clean_labels, all_clean_preds, 
                              target_names=['Чистый', 'Грязный']))
    print("Повреждения:")
    print(classification_report(all_damage_labels, all_damage_preds, 
                              target_names=['Целый', 'Битый']))
    
    # Сохранение финальной модели
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'final_accuracy': best_val_acc,
        'class_weights': {
            'cleanliness': cleanliness_weights.cpu().numpy(),
            'damage': damage_weights.cpu().numpy()
        }
    }, 'final_improved_model.pth')
    
    # Графики обучения
    plot_training_results(history)
    
    # Очистка
    os.remove('train_temp.csv')
    os.remove('val_temp.csv')
    
    print(f"\n🎉 Улучшенное обучение завершено!")
    print(f"✅ Лучшая точность: {best_val_acc:.4f}")
    print(f"✅ Модель сохранена: best_improved_model.pth")
    print(f"✅ График: improved_training_results.png")
    
    return True

def plot_training_results(history):
    """Создание графиков результатов обучения"""
    plt.figure(figsize=(15, 10))
    
    # Loss
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy
    plt.subplot(2, 3, 2)
    plt.plot(history['val_acc'], label='Combined Acc', color='green')
    plt.plot(history['clean_acc'], label='Clean Acc', color='orange')
    plt.plot(history['damage_acc'], label='Damage Acc', color='purple')
    plt.plot(history['overall_acc'], label='Overall Acc', color='brown')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Learning Rate
    plt.subplot(2, 3, 3)
    plt.plot(history['lr'], label='Learning Rate', color='red')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    # Loss comparison
    plt.subplot(2, 3, 4)
    plt.plot(history['train_loss'], label='Train', alpha=0.7)
    plt.plot(history['val_loss'], label='Val', alpha=0.7)
    plt.fill_between(range(len(history['train_loss'])), 
                     history['train_loss'], alpha=0.3)
    plt.fill_between(range(len(history['val_loss'])), 
                     history['val_loss'], alpha=0.3)
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy trends
    plt.subplot(2, 3, 5)
    plt.plot(history['clean_acc'], label='Clean', marker='o', markersize=3)
    plt.plot(history['damage_acc'], label='Damage', marker='s', markersize=3)
    plt.plot(history['overall_acc'], label='Overall', marker='^', markersize=3)
    plt.title('Accuracy Trends')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Final metrics
    plt.subplot(2, 3, 6)
    final_metrics = [
        history['clean_acc'][-1],
        history['damage_acc'][-1], 
        history['overall_acc'][-1],
        history['val_acc'][-1]
    ]
    metric_names = ['Clean', 'Damage', 'Overall', 'Combined']
    bars = plt.bar(metric_names, final_metrics, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.title('Final Metrics')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars, final_metrics):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('improved_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Улучшенные настройки
    config = {
        'epochs': 30,        # Больше эпох
        'batch_size': 16,    # Увеличенный batch size
        'lr': 0.001,         # Learning rate
        'patience': 10       # Early stopping patience
    }
    
    print("⚙️ Улучшенные настройки обучения:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    success = train_improved_model(**config)
    
    if success:
        print(f"\n🚀 Следующие шаги:")
        print(f"1. Проверьте график обучения: improved_training_results.png")
        print(f"2. Обновите app.py для использования улучшенной модели")
        print(f"3. Перезапустите API: python app.py")
    else:
        print(f"\n❌ Обучение не удалось. Проверьте данные.")
