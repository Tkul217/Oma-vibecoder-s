#!/usr/bin/env python3
"""
Подготовка данных из 4 папок: clean_whole, clean_broken, dirty_whole, dirty_broken
"""

import os
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image
import random


def process_four_folders(source_dir='photos'):
    """
    Обработка структуры из 4 папок:
    - clean_whole/ -> чистые целые
    - clean_broken/ -> чистые сломанные
    - dirty_whole/ -> грязные целые
    - dirty_broken/ -> грязные сломанные
    """

    print(f"📁 Обработка папок из {source_dir}/")
    print("=" * 50)

    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"❌ Папка {source_dir} не найдена")
        return False

    # Проверяем наличие нужных папок
    required_folders = ['clean_whole', 'clean_broken', 'dirty_whole', 'dirty_broken']
    missing_folders = []

    for folder in required_folders:
        if not (source_path / folder).exists():
            missing_folders.append(folder)

    if missing_folders:
        print(f"❌ Отсутствующие папки: {missing_folders}")
        return False

    # Создаем структуру для обучения
    create_training_structure()

    # Обрабатываем каждую папку
    dataset_data = []

    print("\n🔄 Обработка изображений...")

    # 1. Папка clean_whole/ -> clean + intact
    clean_whole_folder = source_path / 'clean_whole'
    dataset_data.extend(process_folder(
        clean_whole_folder,
        cleanliness='clean',
        damage='intact',
        category_name='Чистые целые'
    ))

    # 2. Папка clean_broken/ -> clean + damaged
    clean_broken_folder = source_path / 'clean_broken'
    dataset_data.extend(process_folder(
        clean_broken_folder,
        cleanliness='clean',
        damage='damaged',
        category_name='Чистые сломанные'
    ))

    # 3. Папка dirty_whole/ -> dirty + intact
    dirty_whole_folder = source_path / 'dirty_whole'
    dataset_data.extend(process_folder(
        dirty_whole_folder,
        cleanliness='dirty',
        damage='intact',
        category_name='Грязные целые'
    ))

    # 4. Папка dirty_broken/ -> dirty + damaged
    dirty_broken_folder = source_path / 'dirty_broken'
    dataset_data.extend(process_folder(
        dirty_broken_folder,
        cleanliness='dirty',
        damage='damaged',
        category_name='Грязные сломанные'
    ))

    if not dataset_data:
        print("❌ Нет данных для сохранения")
        return False

    # Сохраняем dataset.csv
    df = pd.DataFrame(dataset_data)
    df.to_csv('data/dataset.csv', index=False)

    # Статистика
    print_statistics(df)

    return True


def create_training_structure():
    """Создание структуры для обучения"""

    directories = [
        'data/train/clean_intact',
        'data/train/clean_damaged',
        'data/train/dirty_intact',
        'data/train/dirty_damaged',
        'data/val/clean_intact',
        'data/val/clean_damaged',
        'data/val/dirty_intact',
        'data/val/dirty_damaged'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def process_folder(folder_path, cleanliness, damage, category_name):
    """Обработка одной папки"""

    print(f"\n📂 {category_name}: {folder_path}")

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = []

    # Находим изображения
    for file_path in folder_path.iterdir():
        if file_path.suffix.lower() in image_extensions and file_path.is_file():
            image_files.append(file_path)

    if not image_files:
        print(f"   ⚠️ Изображения не найдены в {folder_path}")
        return []

    print(f"   📸 Найдено изображений: {len(image_files)}")

    dataset_entries = []
    copied_count = 0

    for img_path in image_files:
        try:
            # Проверяем что файл можно открыть
            with Image.open(img_path) as img:
                img.verify()

            # Определяем train/val (80/20)
            split = 'train' if random.random() < 0.8 else 'val'

            # Целевая папка
            target_category = f"{cleanliness}_{damage}"
            target_dir = Path(f"data/{split}/{target_category}")
            target_path = target_dir / img_path.name

            # Копируем файл
            shutil.copy2(img_path, target_path)

            # Добавляем в данные
            dataset_entries.append({
                'image_path': f"{split}/{target_category}/{img_path.name}",
                'cleanliness': cleanliness,
                'damage': damage,
                'split': split,
                'original_path': str(img_path)
            })

            copied_count += 1

        except Exception as e:
            print(f"   ⚠️ Ошибка с {img_path.name}: {e}")
            continue

    print(f"   ✅ Обработано: {copied_count} из {len(image_files)}")
    return dataset_entries


def print_statistics(df):
    """Вывод статистики по датасету"""

    print("\n📊 СТАТИСТИКА ДАТАСЕТА")
    print("=" * 30)
    print(f"Всего изображений: {len(df)}")
    print(f"Обучающая выборка: {len(df[df['split'] == 'train'])}")
    print(f"Валидационная выборка: {len(df[df['split'] == 'val'])}")

    print("\nПо чистоте:")
    clean_count = len(df[df['cleanliness'] == 'clean'])
    dirty_count = len(df[df['cleanliness'] == 'dirty'])
    print(f"  Чистые: {clean_count} ({clean_count / len(df) * 100:.1f}%)")
    print(f"  Грязные: {dirty_count} ({dirty_count / len(df) * 100:.1f}%)")

    print("\nПо повреждениям:")
    intact_count = len(df[df['damage'] == 'intact'])
    damaged_count = len(df[df['damage'] == 'damaged'])
    print(f"  Целые: {intact_count} ({intact_count / len(df) * 100:.1f}%)")
    print(f"  Сломанные: {damaged_count} ({damaged_count / len(df) * 100:.1f}%)")

    print("\nПо категориям:")
    categories = df.groupby(['cleanliness', 'damage']).size()
    for (cleanliness, damage), count in categories.items():
        clean_ru = "Чистые" if cleanliness == 'clean' else "Грязные"
        damage_ru = "целые" if damage == 'intact' else "сломанные"
        print(f"  {clean_ru} {damage_ru}: {count}")

    print(f"\n✅ Данные сохранены: data/dataset.csv")
    print("🚀 Теперь можете запустить обучение: python train.py")


def main():
    print("🚗 inDrive Car Analysis - Подготовка из 4 папок")
    print("=" * 55)

    # Ищем папку с данными
    possible_dirs = ['photos', 'images', 'data_raw', '.']

    source_dir = None
    for dir_name in possible_dirs:
        dir_path = Path(dir_name)
        if (dir_path.exists() and
                (dir_path / 'clean_whole').exists() and
                (dir_path / 'clean_broken').exists() and
                (dir_path / 'dirty_whole').exists() and
                (dir_path / 'dirty_broken').exists()):
            source_dir = dir_name
            break

    if not source_dir:
        print("🔍 Автоматический поиск не удался")
        source_dir = input("Введите путь к папке с подпапками clean_whole/, clean_broken/, dirty_whole/, dirty_broken/: ")
        if not source_dir:
            source_dir = 'photos'

    print(f"📁 Используется папка: {source_dir}")

    # Обрабатываем данные
    success = process_four_folders(source_dir)

    if success:
        print("\n🎉 Подготовка данных завершена успешно!")
        print("\n📋 Следующие шаги:")
        print("1. Проверьте data/dataset.csv")
        print("2. Запустите обучение: python train.py")
        print("3. После обучения модель сохранится в best_model.pth")
        print("4. Обновите app.py чтобы использовать обученную модель")
    else:
        print("\n❌ Ошибка подготовки данных")


if __name__ == "__main__":
    main()
