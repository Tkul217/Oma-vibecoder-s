#!/usr/bin/env python3
"""
Тестирование обученных моделей
"""

import os
import sys
from pathlib import Path

def test_models():
    """Проверка наличия обученных моделей"""
    
    models_dir = Path('models')
    
    print("🔍 Проверка обученных моделей...")
    print(f"📁 Папка моделей: {models_dir.absolute()}")
    
    if not models_dir.exists():
        print("❌ Папка models/ не найдена!")
        return False
    
    # Проверяем наличие моделей
    cleanliness_model = models_dir / 'cleanliness_model.pth'
    condition_model = models_dir / 'condition_model.pth'
    unified_model = models_dir / 'best_unified_model.pth'
    
    models_found = []
    
    if cleanliness_model.exists():
        size = cleanliness_model.stat().st_size / (1024*1024)  # MB
        print(f"✅ cleanliness_model.pth ({size:.1f} MB)")
        models_found.append('cleanliness')
    
    if condition_model.exists():
        size = condition_model.stat().st_size / (1024*1024)  # MB
        print(f"✅ condition_model.pth ({size:.1f} MB)")
        models_found.append('condition')
    
    if unified_model.exists():
        size = unified_model.stat().st_size / (1024*1024)  # MB
        print(f"✅ best_unified_model.pth ({size:.1f} MB)")
        models_found.append('unified')
    
    if not models_found:
        print("❌ Модели не найдены!")
        return False
    
    print(f"\n🎉 Найдено моделей: {len(models_found)}")
    return True

def test_inference():
    """Тест инференса"""
    
    print("\n🧪 Тестирование инференса...")
    
    # Ищем тестовое изображение
    test_images = []
    data_dir = Path('../data')
    
    for folder in ['clean', 'dirty', 'intact', 'damaged']:
        folder_path = data_dir / folder
        if folder_path.exists():
            for ext in ['.jpg', '.jpeg', '.png']:
                images = list(folder_path.glob(f'*{ext}'))
                if images:
                    test_images.extend(images[:1])  # Берем по одному из каждой папки
    
    if not test_images:
        print("❌ Не найдено тестовых изображений!")
        return False
    
    # Тестируем на первом изображении
    test_image = test_images[0]
    print(f"🖼️  Тестовое изображение: {test_image}")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 'analyze.py', str(test_image)
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("✅ Инференс работает!")
            print("📊 Результат:")
            print(result.stdout)
            return True
        else:
            print("❌ Ошибка инференса:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        return False

if __name__ == "__main__":
    print("🚀 ТЕСТИРОВАНИЕ ОБУЧЕННЫХ МОДЕЛЕЙ")
    print("=" * 50)
    
    models_ok = test_models()
    
    if models_ok:
        inference_ok = test_inference()
        
        if inference_ok:
            print("\n🎉 ВСЁ ГОТОВО!")
            print("🌐 Запустите веб-приложение:")
            print("   npm run start:dev")
            print("\n📱 Откройте: http://localhost:5173")
            print("🔮 Загрузите фото автомобиля и протестируйте!")
        else:
            print("\n⚠️  Модели есть, но инференс не работает")
            print("💡 Проверьте файл analyze.py")
    else:
        print("\n❌ Модели не найдены!")
        print("💡 Запустите обучение заново")