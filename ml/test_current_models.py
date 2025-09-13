#!/usr/bin/env python3
"""
Тестирование текущих обученных моделей
"""

import sys
import os
from pathlib import Path
import json

def test_model_loading():
    """Тест загрузки моделей"""
    
    print("🧪 ТЕСТИРОВАНИЕ ТЕКУЩИХ МОДЕЛЕЙ")
    print("=" * 50)
    
    models_dir = Path('models')
    
    # Проверяем наличие моделей
    cleanliness_model = models_dir / 'cleanliness_model.pth'
    condition_model = models_dir / 'condition_model.pth'
    
    print(f"📁 Папка моделей: {models_dir.absolute()}")
    print(f"🧽 Модель чистоты: {'✅' if cleanliness_model.exists() else '❌'}")
    print(f"🔧 Модель повреждений: {'✅' if condition_model.exists() else '❌'}")
    
    if cleanliness_model.exists():
        size = cleanliness_model.stat().st_size / (1024*1024)
        print(f"   Размер: {size:.1f} MB")
    
    if condition_model.exists():
        size = condition_model.stat().st_size / (1024*1024)
        print(f"   Размер: {size:.1f} MB")

def test_analyze_script():
    """Тест скрипта анализа"""
    
    print("\n🔍 ТЕСТИРОВАНИЕ СКРИПТА АНАЛИЗА")
    print("=" * 50)
    
    # Ищем тестовое изображение
    test_images = []
    data_dir = Path('../data')
    
    for folder in ['clean', 'dirty', 'intact', 'damaged']:
        folder_path = data_dir / folder
        if folder_path.exists():
            for ext in ['.jpg', '.jpeg', '.png']:
                images = list(folder_path.glob(f'*{ext}'))
                if images:
                    test_images.extend(images[:1])
    
    if not test_images:
        print("❌ Не найдено тестовых изображений!")
        print("💡 Добавьте изображения в папки data/clean, data/dirty, etc.")
        return
    
    # Тестируем на первом изображении
    test_image = test_images[0]
    print(f"🖼️ Тестовое изображение: {test_image}")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 'analyze.py', str(test_image)
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("✅ Скрипт анализа работает!")
            print("📊 Результат:")
            try:
                data = json.loads(result.stdout)
                print(f"   Чистота: {data.get('cleanliness', 'N/A')} ({data.get('cleanlinessConfidence', 0):.2f})")
                print(f"   Состояние: {data.get('condition', 'N/A')} ({data.get('conditionConfidence', 0):.2f})")
                print(f"   Время: {data.get('processingTime', 0)}ms")
            except:
                print(result.stdout)
        else:
            print("❌ Ошибка скрипта анализа:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")

def check_data_distribution():
    """Проверка распределения данных"""
    
    print("\n📊 АНАЛИЗ ДАННЫХ")
    print("=" * 50)
    
    data_dir = Path('../data')
    
    if not data_dir.exists():
        print("❌ Папка data/ не найдена!")
        return
    
    folders = ['clean', 'dirty', 'intact', 'damaged']
    total_images = 0
    
    for folder in folders:
        folder_path = data_dir / folder
        if folder_path.exists():
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            images = [f for f in folder_path.iterdir() 
                     if f.is_file() and f.suffix.lower() in image_extensions]
            
            count = len(images)
            total_images += count
            
            status = "✅" if count >= 100 else "⚠️" if count > 0 else "❌"
            print(f"  {status} {folder}: {count} изображений")
        else:
            print(f"  ❌ {folder}: папка не найдена")
    
    print(f"\n📈 Всего изображений: {total_images}")
    
    if total_images < 400:
        print("⚠️ Мало данных для качественного обучения!")
        print("💡 Рекомендуется минимум 100 изображений на класс")

def main():
    print("🚗 ДИАГНОСТИКА СИСТЕМЫ АНАЛИЗА АВТОМОБИЛЕЙ")
    print("=" * 60)
    
    test_model_loading()
    check_data_distribution() 
    test_analyze_script()
    
    print("\n" + "=" * 60)
    print("🎯 РЕКОМЕНДАЦИИ:")
    print("1. Если модели есть, но результаты одинаковые - переобучите")
    print("2. Если мало данных - загрузите больше изображений")
    print("3. Используйте: python download_datasets.py")
    print("4. Затем: python train_separate_models.py")

if __name__ == "__main__":
    main()