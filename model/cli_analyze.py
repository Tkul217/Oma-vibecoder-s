#!/usr/bin/env python3
"""
CLI интерфейс для улучшенной модели анализа автомобилей
Используется Node.js сервером для анализа изображений
"""

import sys
import json
import argparse
import os
from pathlib import Path

# Добавляем текущую папку в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_car_model import AdvancedCarAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Анализ изображения автомобиля с помощью улучшенной модели')
    parser.add_argument('--analyze', type=str, help='Путь к изображению для анализа')
    parser.add_argument('image_path', nargs='?', help='Путь к изображению для анализа')
    
    args = parser.parse_args()
    
    # Определяем путь к изображению
    image_path = args.analyze or args.image_path
    
    if not image_path:
        error_result = {
            'success': False,
            'error': 'Не указан путь к изображению. Используйте: python cli_analyze.py <путь_к_изображению>'
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)
    
    if not os.path.exists(image_path):
        error_result = {
            'success': False,
            'error': f'Файл не найден: {image_path}'
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)
    
    try:
        # Инициализируем анализатор
        model_path = 'best_improved_model.pth' if os.path.exists('best_improved_model.pth') else 'final_improved_model.pth'
        analyzer = AdvancedCarAnalyzer(model_path=model_path, confidence_threshold=0.7)
        
        # Загружаем и анализируем изображение
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        # Анализируем изображение
        result = analyzer.analyze_image(image, use_ensemble=True)
        
        # Добавляем информацию об успехе
        result['success'] = True
        result['filename'] = os.path.basename(image_path)
        
        # Выводим результат в JSON формате
        print(json.dumps(result, ensure_ascii=False))
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': f'Ошибка анализа: {str(e)}'
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

if __name__ == '__main__':
    main()
