#!/usr/bin/env python3
"""
Интеграция с Roboflow API для анализа автомобилей
Использует готовую обученную модель вместо локального обучения
"""

import requests
import json
import sys
import time
import base64
from pathlib import Path

class RoboflowCarAnalyzer:
    def __init__(self):
        # API настройки из скриншота
        self.api_url = "https://serverless.roboflow.com"
        self.api_key = "oKaglUXuEl3b4O3SaWLY"
        self.model_id = "car-scratch-and-dent/3"

    def analyze_image(self, image_path):
        """Анализ изображения через Roboflow API"""
        start_time = time.time()

        try:
            # Читаем изображение
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # Кодируем в base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')

            # Отправляем запрос к API
            response = requests.post(
                f"{self.api_url}/{self.model_id}",
                params={"api_key": self.api_key},
                json={"image": image_b64},
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                return self._process_roboflow_result(result, start_time)
            else:
                raise Exception(f"API error: {response.status_code} - {response.text}")

        except Exception as e:
            return {
                'error': str(e),
                'processingTime': int((time.time() - start_time) * 1000)
            }

    def _process_roboflow_result(self, result, start_time):
        """Обработка результата от Roboflow API"""

        processing_time = int((time.time() - start_time) * 1000)

        # Анализируем предсказания
        predictions = result.get('predictions', [])

        # Определяем наличие повреждений
        has_damage = len(predictions) > 0
        damage_confidence = 0.0

        if has_damage:
            # Берем максимальную уверенность среди всех предсказаний
            damage_confidence = max(pred.get('confidence', 0) for pred in predictions)
        else:
            # Если нет предсказаний, значит машина целая
            damage_confidence = 0.85  # Высокая уверенность в отсутствии повреждений

        # Определяем чистоту (пока простая эвристика)
        # В будущем можно добавить отдельную модель для чистоты
        filename = Path(image_path).name.lower()
        is_dirty = any(word in filename for word in ['dirty', 'mud', 'dust', 'soil'])

        cleanliness = 'dirty' if is_dirty else 'clean'
        cleanliness_confidence = 0.75  # Средняя уверенность для эвристики

        condition = 'damaged' if has_damage else 'intact'
        condition_confidence = damage_confidence if has_damage else (1.0 - damage_confidence)

        return {
            'cleanliness': cleanliness,
            'cleanlinessConfidence': cleanliness_confidence,
            'condition': condition,
            'conditionConfidence': condition_confidence,
            'processingTime': processing_time,
            'roboflow_predictions': predictions,  # Дополнительная информация
            'damage_count': len(predictions)
        }

def main():
    """CLI интерфейс"""
    if len(sys.argv) != 2:
        print(json.dumps({'error': 'Usage: python roboflow_inference.py <image_path>'}))
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(json.dumps({'error': f'Image file not found: {image_path}'}))
        sys.exit(1)

    try:
        analyzer = RoboflowCarAnalyzer()
        result = analyzer.analyze_image(image_path)
        print(json.dumps(result, ensure_ascii=False))

        # Успешный выход
        sys.exit(0 if 'error' not in result else 1)

    except Exception as e:
        error_result = {
            'error': f'Unexpected error: {str(e)}',
            'processingTime': 0
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == '__main__':
    main()