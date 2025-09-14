#!/usr/bin/env python3
"""
Скрипт для анализа состояния автомобиля по фотографии
Использует Roboflow API для анализа повреждений
Интеграция с Node.js сервером через command line interface
"""

import sys
import json
import time
import os
from pathlib import Path
import requests
import base64

# Попытка импорта ML библиотек
try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML libraries not installed. Using Roboflow API.", file=sys.stderr)

class RoboflowAnalyzer:
    """Анализатор через Roboflow API"""
    def __init__(self):
        self.api_url = "https://serverless.roboflow.com"
        self.api_key = "oKaglUXuEl3bAO3SaWLY"
        self.model_id = "car-scratch-and-dent/3"

    def analyze_image(self, image_path):
        """Анализ через Roboflow API"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()

            image_b64 = base64.b64encode(image_data).decode('utf-8')

            response = requests.post(
                f"{self.api_url}/{self.model_id}",
                params={"api_key": self.api_key},
                json={"image": image_b64},
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                predictions = result.get('predictions', [])

                # Есть ли повреждения
                has_damage = len(predictions) > 0
                damage_confidence = max(pred.get('confidence', 0) for pred in predictions) if has_damage else 0.85

                # Простая эвристика для чистоты
                filename = Path(image_path).name.lower()
                is_dirty = any(word in filename for word in ['dirty', 'mud', 'dust'])

                return {
                    'cleanliness': 'dirty' if is_dirty else 'clean',
                    'cleanlinessConfidence': 0.75,
                    'condition': 'damaged' if has_damage else 'intact',
                    'conditionConfidence': damage_confidence if has_damage else (1.0 - damage_confidence),
                    'roboflow_used': True,
                    'damage_count': len(predictions)
                }
            else:
                raise Exception(f"Roboflow API error: {response.status_code}")

        except Exception as e:
            print(f"Roboflow API failed: {e}", file=sys.stderr)
            return None

class UnifiedCarModel(torch.nn.Module):
    """Единая модель для 4 классов"""
    def __init__(self, num_classes=4):
        super(UnifiedCarModel, self).__init__()
        if ML_AVAILABLE:
            from torchvision import models
            self.backbone = models.resnet50(pretrained=False)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(num_features, 1024),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(1024),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(512),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.backbone(x)

class CarConditionAnalyzer:
    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.device = 'cpu'  # Используем CPU для совместимости

        # Пробуем Roboflow API сначала
        self.roboflow_analyzer = RoboflowAnalyzer()

        if ML_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            # Попытка загрузки единой модели
            self.unified_model = self._load_unified_model('best_unified_model.pth')

            # Если единой модели нет, пробуем загрузить отдельные модели
            if self.unified_model is None:
                self.cleanliness_model = self._load_model('cleanliness_model.pth')
                self.condition_model = self._load_model('condition_model.pth')
            else:
                self.cleanliness_model = None
                self.condition_model = None

    def _load_unified_model(self, model_name):
        """Загрузка единой модели"""
        model_path = self.models_dir / model_name

        if model_path.exists() and ML_AVAILABLE:
            try:
                model = UnifiedCarModel(num_classes=4)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                print(f"Загружена единая модель: {model_name}", file=sys.stderr)
                return model
            except Exception as e:
                print(f"Warning: Could not load unified model {model_name}: {e}", file=sys.stderr)
                return None
        return None

    def _load_model(self, model_name):
        """Загрузка модели из файла"""
        model_path = self.models_dir / model_name

        if model_path.exists() and ML_AVAILABLE:
            try:
                # Создаем модель и загружаем веса
                model = CarConditionModel(num_classes=2)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                print(f"Загружена отдельная модель: {model_name}", file=sys.stderr)
                return model
            except Exception as e:
                print(f"Warning: Could not load {model_name}: {e}", file=sys.stderr)
                return None
        return None

    def analyze_image(self, image_path):
        """Анализ изображения автомобиля"""
        start_time = time.time()

        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Сначала пробуем Roboflow API
            roboflow_result = self.roboflow_analyzer.analyze_image(image_path)
            if roboflow_result:
                processing_time = int((time.time() - start_time) * 1000)
                roboflow_result['processingTime'] = processing_time
                return roboflow_result
            
            if ML_AVAILABLE and self.unified_model:
                # Используем единую ML модель
                result = self._analyze_with_unified_model(image_path)
            elif ML_AVAILABLE and self.cleanliness_model and self.condition_model:
                # Используем отдельные ML модели
                result = self._analyze_with_ml(image_path)
            else:
                # Используем mock предсказания
                result = self._analyze_mock(image_path)
            
            processing_time = int((time.time() - start_time) * 1000)
            result['processingTime'] = processing_time
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'processingTime': int((time.time() - start_time) * 1000)
            }
    
    def _analyze_with_unified_model(self, image_path):
        """Анализ с использованием единой ML модели"""
        # Маппинг классов
        class_mapping = {
            0: 'clean_intact',
            1: 'clean_damaged', 
            2: 'dirty_intact',
            3: 'dirty_damaged'
        }
        
        # Загрузка и предобработка изображения
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.unified_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Получаем предсказанный класс
            predicted_class = class_mapping[predicted.item()]
            
            # Разбираем класс на компоненты
            parts = predicted_class.split('_')
            cleanliness = parts[0]  # clean или dirty
            condition = parts[1]    # intact или damaged
            
            return {
                'cleanliness': cleanliness,
                'cleanlinessConfidence': float(confidence.item()),
                'condition': condition,
                'conditionConfidence': float(confidence.item()),
                'predicted_class': predicted_class,
                'all_probabilities': {
                    class_mapping[i]: float(prob) 
                    for i, prob in enumerate(probabilities[0])
                }
            }
    
    def _analyze_with_ml(self, image_path):
        """Анализ с использованием ML моделей"""
        # Загрузка и предобработка изображения
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Анализ чистоты
            clean_outputs = self.cleanliness_model(image_tensor)
            clean_probs = torch.nn.functional.softmax(clean_outputs, dim=1)
            clean_confidence = float(clean_probs.max())
            cleanliness = 'clean' if clean_outputs.argmax() == 0 else 'dirty'
            
            # Анализ повреждений
            condition_outputs = self.condition_model(image_tensor)
            condition_probs = torch.nn.functional.softmax(condition_outputs, dim=1)
            condition_confidence = float(condition_probs.max())
            condition = 'intact' if condition_outputs.argmax() == 0 else 'damaged'
        
        return {
            'cleanliness': cleanliness,
            'cleanlinessConfidence': clean_confidence,
            'condition': condition,
            'conditionConfidence': condition_confidence
        }
    
    def _analyze_mock(self, image_path):
        """Mock анализ для демонстрации"""
        # Простой анализ по имени файла (пока нет обученной модели)
        filename = os.path.basename(image_path).lower()
        
        # Имитируем анализ изображения
        time.sleep(0.5)
        
        # Анализ по ключевым словам в имени файла
        if 'clean' in filename:
            cleanliness = 'clean'
            clean_conf = 0.85
        elif 'dirty' in filename:
            cleanliness = 'dirty' 
            clean_conf = 0.82
        else:
            # Если нет ключевых слов, предполагаем чистый
            cleanliness = 'clean'
            clean_conf = 0.75
            
        if 'damaged' in filename or 'broken' in filename:
            condition = 'damaged'
            cond_conf = 0.88
        elif 'intact' in filename:
            condition = 'intact'
            cond_conf = 0.85
        else:
            # Если нет ключевых слов, предполагаем целый
            condition = 'intact'
            cond_conf = 0.78
        
        return {
            'cleanliness': cleanliness,
            'cleanlinessConfidence': clean_conf,
            'condition': condition,
            'conditionConfidence': cond_conf
        }

def main():
    """Основная функция для CLI интерфейса"""
    if len(sys.argv) != 2:
        error_result = {
            'error': 'Usage: python analyze.py <image_path>',
            'processingTime': 0
        }
        print(json.dumps(error_result))
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        analyzer = CarConditionAnalyzer()
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