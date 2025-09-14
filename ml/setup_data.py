#!/usr/bin/env python3
"""
Скрипт для проверки и настройки данных
"""

import os
from pathlib import Path

def check_data_structure():
    """Проверка структуры данных"""

    print("🔍 Проверка структуры данных...")

    data_dir = Path('../data')
    required_folders = ['clean_intact', 'clean_damaged', 'dirty_intact', 'dirty_damaged']

    print(f"📁 Базовая папка: {data_dir.absolute()}")

    if not data_dir.exists():
        print("❌ Папка 'data' не найдена!")
        return False

    total_images = 0
    all_good = True

    for folder in required_folders:
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

            if count == 0:
                print(f"      💡 Добавьте изображения в data/{folder}/")
        else:
            print(f"  ❌ {folder}: папка не найдена!")
            all_good = False

    print(f"\n📊 Всего изображений: {total_images}")

    if total_images >= 200:
        print("🎉 Достаточно данных для обучения!")
        print("🚀 Запустите: python train_unified_model.py")
        return True
    elif total_images > 0:
        print("⚠️  Мало данных. Рекомендуется минимум 200 изображений")
        print("💡 Добавьте больше изображений и попробуйте снова")
        return False
    else:
        print("❌ Нет данных для обучения!")
        print("📋 Инструкция:")
        print("1. Добавьте изображения в папки data/")
        print("2. Минимум 50 изображений на папку")
        print("3. Запустите: python train_unified_model.py")
        return False

if __name__ == "__main__":
    check_data_structure()