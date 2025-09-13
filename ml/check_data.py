#!/usr/bin/env python3
"""
Скрипт для проверки структуры данных
"""

import os
from pathlib import Path

def check_data_structure():
    """Проверка структуры папок с данными"""
    
    data_dir = Path('../data')  # Путь относительно папки ml
    required_folders = ['clean', 'dirty', 'intact', 'damaged']
    
    print("🔍 Проверка структуры данных...")
    print(f"📁 Базовая папка: {data_dir.absolute()}")
    
    if not data_dir.exists():
        print("❌ Папка 'data' не найдена!")
        return False
    
    all_good = True
    
    for folder in required_folders:
        folder_path = data_dir / folder
        
        if folder_path.exists():
            # Подсчет изображений
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            images = [f for f in folder_path.iterdir() 
                     if f.is_file() and f.suffix.lower() in image_extensions]
            
            print(f"✅ {folder}/: {len(images)} изображений")
            
            if len(images) == 0:
                print(f"⚠️  Папка {folder}/ пустая!")
        else:
            print(f"❌ Папка {folder}/ не найдена!")
            all_good = False
    
    if all_good:
        print("\n🎉 Структура данных корректна!")
        print("\n📋 Следующие шаги:")
        print("1. Разложите фотографии по папкам:")
        print("   - data/clean/ - чистые машины")
        print("   - data/dirty/ - грязные машины") 
        print("   - data/intact/ - целые машины")
        print("   - data/damaged/ - битые машины")
        print("2. Запустите: python train_separate_models.py")
    else:
        print("\n❌ Исправьте структуру папок и попробуйте снова")
    
    return all_good

if __name__ == "__main__":
    check_data_structure()