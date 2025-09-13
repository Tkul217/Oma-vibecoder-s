#!/usr/bin/env python3
"""
Скрипт для загрузки и подготовки датасетов для обучения модели анализа автомобилей
Использует датасеты с Roboflow для царапин, вмятин и ржавчины
"""

import os
import requests
import zipfile
import shutil
from pathlib import Path
import json
from PIL import Image
import cv2
import numpy as np

def download_file(url, filename):
    """Загрузка файла с прогресс-баром"""
    print(f"📥 Загружаем {filename}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r📊 {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
    
    print(f"\n✅ {filename} загружен!")

def setup_roboflow_dataset():
    """Настройка для загрузки с Roboflow"""
    
    print("🤖 НАСТРОЙКА ROBOFLOW ДАТАСЕТОВ")
    print("=" * 50)
    
    print("""
📋 Для загрузки датасетов нужно:

1. Зарегистрироваться на https://roboflow.com
2. Получить API ключ в настройках профиля
3. Выбрать один из датасетов:
   
   🔧 Царапины и вмятины:
   - https://universe.roboflow.com/seva-at1qy/rust-and-scrach
   - https://universe.roboflow.com/carpro/car-scratch-and-dent
   - https://universe.roboflow.com/project-kmnth/car-scratch-xgxzs

📝 Инструкция:
1. Откройте один из датасетов
2. Нажмите "Download Dataset"
3. Выберите формат "Folder Structure"
4. Скопируйте команду загрузки
5. Запустите её в папке ml/

Пример команды:
curl -L "https://app.roboflow.com/ds/..." > dataset.zip
""")

def create_manual_structure():
    """Создание структуры для ручной загрузки"""
    
    print("📁 Создаем структуру папок для ручной загрузки...")
    
    # Создаем папки для данных
    data_dir = Path('../data')
    data_dir.mkdir(exist_ok=True)
    
    folders = {
        'clean': 'Чистые автомобили без повреждений',
        'dirty': 'Грязные автомобили', 
        'intact': 'Автомобили без царапин и вмятин',
        'damaged': 'Автомобили с царапинами, вмятинами, ржавчиной'
    }
    
    for folder, description in folders.items():
        folder_path = data_dir / folder
        folder_path.mkdir(exist_ok=True)
        
        # Создаем README в каждой папке
        readme_path = folder_path / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"""# {folder.title()}

{description}

## Требования к изображениям:
- Форматы: JPG, PNG
- Минимум: 100 изображений
- Рекомендуется: 500+ изображений
- Разрешение: любое (будет изменено до 224x224)

## Источники изображений:
- Roboflow датасеты
- Google Images
- Собственные фотографии
- Kaggle датасеты

## Примеры поиска:
- "car scratches dataset"
- "vehicle damage detection"
- "automotive condition assessment"
""")
    
    print("✅ Структура папок создана!")
    print(f"📁 Путь: {data_dir.absolute()}")

def convert_roboflow_to_folders(dataset_path):
    """Конвертация Roboflow датасета в папки по классам"""
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Датасет не найден: {dataset_path}")
        return
    
    print(f"🔄 Конвертируем датасет: {dataset_path}")
    
    # Ищем аннотации
    annotations_files = list(dataset_path.glob('**/*.json'))
    
    if not annotations_files:
        print("❌ Не найдены JSON аннотации!")
        return
    
    # Создаем папки для классов
    output_dir = Path('../data')
    output_dir.mkdir(exist_ok=True)
    
    class_folders = {
        'clean': output_dir / 'clean',
        'dirty': output_dir / 'dirty', 
        'intact': output_dir / 'intact',
        'damaged': output_dir / 'damaged'
    }
    
    for folder in class_folders.values():
        folder.mkdir(exist_ok=True)
    
    # Обрабатываем аннотации
    processed = 0
    
    for ann_file in annotations_files:
        try:
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            # Ищем соответствующее изображение
            image_name = ann_file.stem + '.jpg'
            image_path = ann_file.parent / image_name
            
            if not image_path.exists():
                image_name = ann_file.stem + '.png'
                image_path = ann_file.parent / image_name
            
            if not image_path.exists():
                continue
            
            # Определяем класс по аннотациям
            has_damage = False
            
            if 'annotations' in data:
                for annotation in data['annotations']:
                    if 'category_name' in annotation:
                        category = annotation['category_name'].lower()
                        if any(word in category for word in ['scratch', 'dent', 'rust', 'damage']):
                            has_damage = True
                            break
            
            # Копируем в соответствующую папку
            if has_damage:
                dest_folder = class_folders['damaged']
            else:
                dest_folder = class_folders['intact']
            
            dest_path = dest_folder / f"{processed:04d}_{image_path.name}"
            shutil.copy2(image_path, dest_path)
            
            processed += 1
            
            if processed % 100 == 0:
                print(f"📊 Обработано: {processed} изображений")
                
        except Exception as e:
            print(f"⚠️ Ошибка обработки {ann_file}: {e}")
    
    print(f"✅ Конвертация завершена! Обработано: {processed} изображений")

def download_sample_images():
    """Загрузка примеров изображений для тестирования"""
    
    print("🖼️ Загружаем примеры изображений...")
    
    # Примеры URL изображений (замените на реальные)
    sample_urls = {
        'clean': [
            'https://example.com/clean_car1.jpg',
            'https://example.com/clean_car2.jpg',
        ],
        'damaged': [
            'https://example.com/damaged_car1.jpg', 
            'https://example.com/damaged_car2.jpg',
        ]
    }
    
    data_dir = Path('../data')
    
    for category, urls in sample_urls.items():
        category_dir = data_dir / category
        category_dir.mkdir(exist_ok=True)
        
        for i, url in enumerate(urls):
            try:
                filename = category_dir / f"sample_{i+1}.jpg"
                download_file(url, filename)
            except Exception as e:
                print(f"⚠️ Не удалось загрузить {url}: {e}")

def main():
    """Основная функция"""
    
    print("🚗 ЗАГРУЗЧИК ДАТАСЕТОВ ДЛЯ АНАЛИЗА АВТОМОБИЛЕЙ")
    print("=" * 60)
    
    print("""
Выберите способ подготовки данных:

1. 🤖 Настройка Roboflow (рекомендуется)
2. 📁 Создать структуру для ручной загрузки  
3. 🔄 Конвертировать существующий Roboflow датасет
4. 🖼️ Загрузить примеры изображений

Введите номер (1-4): """, end='')
    
    try:
        choice = input().strip()
        
        if choice == '1':
            setup_roboflow_dataset()
        elif choice == '2':
            create_manual_structure()
        elif choice == '3':
            dataset_path = input("Путь к Roboflow датасету: ").strip()
            convert_roboflow_to_folders(dataset_path)
        elif choice == '4':
            download_sample_images()
        else:
            print("❌ Неверный выбор!")
            
    except KeyboardInterrupt:
        print("\n👋 Отменено пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()