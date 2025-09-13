#!/usr/bin/env python3
"""
Скрипт для настройки обучения реальной ML модели
"""

import os
import shutil
from pathlib import Path

def create_data_structure():
    """Создание структуры папок для данных"""
    
    print("🏗️  Создание структуры папок для обучения...")
    
    # Создаем базовую папку data
    data_dir = Path('../data')
    data_dir.mkdir(exist_ok=True)
    
    # Создаем папки для 4 классов
    folders = [
        'clean_intact',     # чистые целые машины
        'clean_damaged',    # чистые битые машины  
        'dirty_intact',     # грязные целые машины
        'dirty_damaged'     # грязные битые машины
    ]
    
    for folder in folders:
        folder_path = data_dir / folder
        folder_path.mkdir(exist_ok=True)
        
        # Создаем README в каждой папке
        readme_path = folder_path / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"""# {folder.replace('_', ' ').title()}

Поместите сюда изображения автомобилей категории: **{folder.replace('_', ' ')}**

## Требования к изображениям:
- Форматы: JPG, PNG, BMP, TIFF
- Минимум: 50 изображений
- Рекомендуется: 200+ изображений
- Качество: чем выше, тем лучше

## Примеры:
- Хорошо видимый автомобиль
- Разные ракурсы (спереди, сбоку, сзади)
- Разное освещение
- Разные модели автомобилей
""")
    
    print("✅ Структура папок создана!")
    print(f"📁 Путь: {data_dir.absolute()}")
    print("\n📋 Следующие шаги:")
    print("1. Разложите ваши фотографии по папкам:")
    for folder in folders:
        print(f"   - data/{folder}/ - {folder.replace('_', ' ')}")
    print("2. Запустите: python train_unified_model.py")

def check_data():
    """Проверка наличия данных"""
    
    data_dir = Path('../data')
    
    if not data_dir.exists():
        print("❌ Папка data/ не найдена!")
        return False
    
    folders = ['clean_intact', 'clean_damaged', 'dirty_intact', 'dirty_damaged']
    total_images = 0
    
    print("🔍 Проверка данных:")
    
    for folder in folders:
        folder_path = data_dir / folder
        if folder_path.exists():
            # Подсчет изображений
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            images = [f for f in folder_path.iterdir() 
                     if f.is_file() and f.suffix.lower() in image_extensions]
            
            count = len(images)
            total_images += count
            
            status = "✅" if count >= 50 else "⚠️" if count > 0 else "❌"
            print(f"  {status} {folder}: {count} изображений")
        else:
            print(f"  ❌ {folder}: папка не найдена")
    
    print(f"\n📊 Всего изображений: {total_images}")
    
    if total_images >= 200:
        print("🎉 Достаточно данных для обучения!")
        return True
    elif total_images > 0:
        print("⚠️  Мало данных. Рекомендуется минимум 200 изображений (50 на класс)")
        return True
    else:
        print("❌ Нет данных для обучения!")
        return False

def main():
    print("🚀 НАСТРОЙКА ОБУЧЕНИЯ ML МОДЕЛИ")
    print("=" * 50)
    
    # Создаем структуру папок
    create_data_structure()
    
    print("\n" + "=" * 50)
    
    # Проверяем данные
    has_data = check_data()
    
    print("\n" + "=" * 50)
    print("📝 ИНСТРУКЦИЯ:")
    print("1. Разложите фотографии автомобилей по папкам data/")
    print("2. Запустите: python train_unified_model.py")
    print("3. Дождитесь завершения обучения")
    print("4. Модель автоматически заменит mock предсказания")

if __name__ == "__main__":
    main()