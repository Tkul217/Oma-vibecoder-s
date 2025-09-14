#!/usr/bin/env python3
"""
Простое обучение модели на 3 типах изображений
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
from car_model import CarConditionClassifier

class SimpleCarDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        
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
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        # Метки (0/1)
        cleanliness_label = 0 if row['cleanliness'] == 'clean' else 1
        damage_label = 0 if row['damage'] == 'intact' else 1
        
        return {
            'image': image,
            'cleanliness': torch.tensor(cleanliness_label, dtype=torch.long),
            'damage': torch.tensor(damage_label, dtype=torch.long)
        }

def train_simple_model(epochs=15, batch_size=8, lr=0.001):
    """Простое обучение модели"""
    
    print("🚗 Обучение модели анализа автомобилей")
    print("=" * 45)
    
    # Проверяем наличие данных
    if not os.path.exists('data/dataset.csv'):
        print("❌ Файл data/dataset.csv не найден!")
        print("💡 Сначала запустите: python prepare_3folders.py")
        return False
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Устройство: {device}")
    
    # Трансформации
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Загрузка данных
    print("\n📊 Подготовка данных...")
    
    df = pd.read_csv('data/dataset.csv')
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    
    print(f"   Обучающих: {len(train_df)}")
    print(f"   Валидационных: {len(val_df)}")
    
    if len(train_df) < 4:
        print("⚠️ Очень мало обучающих данных! Рекомендуется минимум 20-50 изображений")
    
    # Временные CSV файлы
    train_df.to_csv('train_temp.csv', index=False)
    val_df.to_csv('val_temp.csv', index=False)
    
    # Датасеты
    train_dataset = SimpleCarDataset('train_temp.csv', 'data', train_transform)
    val_dataset = SimpleCarDataset('val_temp.csv', 'data', val_transform) 
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Модель
    print("\n🤖 Создание модели...")
    model = CarConditionClassifier()
    model.to(device)
    
    # Критерии и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # История
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    print(f"\n🎯 Начало обучения на {epochs} эпох...")
    print("=" * 45)
    
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
            
            optimizer.zero_grad()
            
            # Прямой проход
            clean_logits, damage_logits = model(images)
            
            # Loss (комбинированный)
            clean_loss = criterion(clean_logits, clean_labels)
            damage_loss = criterion(damage_logits, damage_labels)
            total_loss = clean_loss + damage_loss
            
            # Обратный проход
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_batches += 1
            
            pbar.set_postfix({'loss': total_loss.item()})
        
        # ВАЛИДАЦИЯ
        model.eval()
        val_loss = 0.0
        correct_clean = 0
        correct_damage = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Валидация")
            for batch in pbar:
                images = batch['image'].to(device)
                clean_labels = batch['cleanliness'].to(device)
                damage_labels = batch['damage'].to(device)
                
                clean_logits, damage_logits = model(images)
                
                # Loss
                clean_loss = criterion(clean_logits, clean_labels)
                damage_loss = criterion(damage_logits, damage_labels)
                total_loss = clean_loss + damage_loss
                val_loss += total_loss.item()
                
                # Точность
                _, clean_pred = torch.max(clean_logits, 1)
                _, damage_pred = torch.max(damage_logits, 1)
                
                correct_clean += (clean_pred == clean_labels).sum().item()
                correct_damage += (damage_pred == damage_labels).sum().item()
                total += images.size(0)
        
        # Метрики
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / len(val_loader)
        clean_acc = correct_clean / total
        damage_acc = correct_damage / total
        combined_acc = (clean_acc + damage_acc) / 2
        
        # Сохранение истории
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(combined_acc)
        
        # Вывод результатов
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Clean Acc: {clean_acc:.4f} | Damage Acc: {damage_acc:.4f} | Combined: {combined_acc:.4f}")
        
        # Сохранение лучшей модели
        if combined_acc > best_val_acc:
            best_val_acc = combined_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': combined_acc,
                'clean_acc': clean_acc,
                'damage_acc': damage_acc
            }, 'best_model.pth')
            print(f"✅ Сохранена лучшая модель! Точность: {combined_acc:.4f}")
    
    # Сохранение финальной модели
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'final_accuracy': best_val_acc
    }, 'final_model.pth')
    
    # График обучения
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()
    
    # Очистка
    os.remove('train_temp.csv')
    os.remove('val_temp.csv')
    
    print(f"\n🎉 Обучение завершено!")
    print(f"✅ Лучшая точность: {best_val_acc:.4f}")
    print(f"✅ Модель сохранена: best_model.pth")
    print(f"✅ График: training_results.png")
    
    return True

if __name__ == "__main__":
    # Простые настройки
    config = {
        'epochs': 15,        # Количество эпох  
        'batch_size': 8,     # Размер батча (уменьшено для малых данных)
        'lr': 0.001         # Learning rate
    }
    
    print("⚙️ Настройки обучения:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    success = train_simple_model(**config)
    
    if success:
        print(f"\n🚀 Следующие шаги:")
        print(f"1. Проверьте график обучения: training_results.png")
        print(f"2. Обновите app.py для использования обученной модели")
        print(f"3. Перезапустите API: python app.py")
    else:
        print(f"\n❌ Обучение не удалось. Проверьте данные.")