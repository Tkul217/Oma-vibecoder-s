import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import random
import os

class CarConditionClassifier(nn.Module):
    def __init__(self):
        super(CarConditionClassifier, self).__init__()
        
        # Используем ResNet50 как backbone
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Классификатор чистоты (чистый/грязный)
        self.cleanliness_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
        # Классификатор повреждений (целый/битый)
        self.damage_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        cleanliness_logits = self.cleanliness_head(features)
        damage_logits = self.damage_head(features)
        return cleanliness_logits, damage_logits

class CarAnalyzer:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CarConditionClassifier()
        self.model.to(self.device)
        
        # Загружаем обученную модель если есть
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                print(f"✅ Обученная модель загружена: {model_path}")
                self.trained = True
            except Exception as e:
                print(f"⚠️ Ошибка загрузки модели: {e}")
                print("🔄 Используется необученная модель (демо режим)")
                self.model.eval()
                self.trained = False
        else:
            print("🔄 Обученная модель не найдена, используется демо режим")
            self.model.eval()
            self.trained = False
        
        # Трансформации для изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = {
            'cleanliness': ['Чистый', 'Грязный'],
            'damage': ['Целый', 'Битый']
        }
    
    def analyze_image(self, image):
        """Анализ изображения автомобиля"""
        try:
            # Предобработка
            if isinstance(image, torch.Tensor):
                processed_image = image.unsqueeze(0)
            else:
                processed_image = self.transform(image).unsqueeze(0)
            
            processed_image = processed_image.to(self.device)
            
            # Если модель обучена - используем её
            if self.trained:
                with torch.no_grad():
                    cleanliness_logits, damage_logits = self.model(processed_image)
                    
                    # Получаем вероятности
                    cleanliness_probs = torch.softmax(cleanliness_logits, dim=1)
                    damage_probs = torch.softmax(damage_logits, dim=1)
                    
                    # Получаем предсказания
                    cleanliness_pred = torch.argmax(cleanliness_probs, dim=1).item()
                    damage_pred = torch.argmax(damage_probs, dim=1).item()
                    
                    # Уверенность
                    cleanliness_conf = float(cleanliness_probs[0][cleanliness_pred])
                    damage_conf = float(damage_probs[0][damage_pred])
                    
                    # Классы
                    cleanliness_class = self.class_names['cleanliness'][cleanliness_pred]
                    damage_class = self.class_names['damage'][damage_pred]
                    
                    return self._format_result(cleanliness_class, damage_class, 
                                             cleanliness_conf, damage_conf, trained=True)
            
            # Иначе демо режим
            return self._generate_demo_result()
            
        except Exception as e:
            raise Exception(f"Ошибка анализа: {e}")
    
    def _format_result(self, cleanliness_class, damage_class, clean_conf, damage_conf, trained=False):
        """Форматирование результата"""
        
        # Определяем общее состояние
        if cleanliness_class == 'Чистый' and damage_class == 'Целый':
            overall = 'Отличное'
        elif cleanliness_class == 'Грязный' and damage_class == 'Целый':
            overall = 'Хорошее (требует мойки)'
        elif cleanliness_class == 'Чистый' and damage_class == 'Битый':
            overall = 'Требует ремонта'
        else:
            overall = 'Плохое (требует ремонта и мойки)'
        
        # Рекомендации
        recommendations = []
        if cleanliness_class == 'Грязный':
            if clean_conf > 0.8:
                recommendations.append('Автомобиль определенно нуждается в мойке')
            else:
                recommendations.append('Рекомендуется помыть автомобиль')
            recommendations.append('Чистый автомобиль повышает рейтинг водителя')
        
        if damage_class == 'Битый':
            if damage_conf > 0.8:
                recommendations.append('Обнаружены серьезные повреждения')
                recommendations.append('Требуется техосмотр и ремонт')
            else:
                recommendations.append('Возможны повреждения - рекомендуется осмотр')
            recommendations.append('Уведомите пассажира о состоянии автомобиля')
            recommendations.append('Обратитесь в техподдержку inDrive')
        
        if not recommendations:
            recommendations.append('Автомобиль в отличном состоянии!')
            recommendations.append('Поддерживайте такое состояние для высокого рейтинга')
        
        return {
            'cleanliness': {
                'class': cleanliness_class,
                'confidence': clean_conf
            },
            'damage': {
                'class': damage_class,
                'confidence': damage_conf
            },
            'overall_condition': overall,
            'recommendations': recommendations,
            'model_trained': trained
        }
    
    def _generate_demo_result(self):
        """Генерация демо-результатов"""
        # Реалистичные сценарии
        scenarios = [
            {'clean': 0, 'damage': 0, 'clean_conf': 0.92, 'damage_conf': 0.89},
            {'clean': 1, 'damage': 0, 'clean_conf': 0.87, 'damage_conf': 0.94},
            {'clean': 0, 'damage': 1, 'clean_conf': 0.76, 'damage_conf': 0.82},
            {'clean': 1, 'damage': 1, 'clean_conf': 0.91, 'damage_conf': 0.78},
        ]
        
        scenario = random.choice(scenarios)
        
        # Добавляем немного случайности к уверенности
        clean_conf = max(0.6, min(0.99, scenario['clean_conf'] + random.uniform(-0.05, 0.05)))
        damage_conf = max(0.6, min(0.99, scenario['damage_conf'] + random.uniform(-0.05, 0.05)))
        
        cleanliness_class = self.class_names['cleanliness'][scenario['clean']]
        damage_class = self.class_names['damage'][scenario['damage']]
        
        return self._format_result(cleanliness_class, damage_class, 
                                 clean_conf, damage_conf, trained=False)