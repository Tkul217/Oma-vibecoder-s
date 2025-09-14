#!/usr/bin/env python3
"""
Улучшенная модель для анализа автомобилей с повышенной точностью
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import random
import os
import cv2
from typing import Dict, List, Tuple, Optional
import torchvision.transforms.functional as TF
from torch.nn import MultiheadAttention

class SpatialAttention(nn.Module):
    """Spatial Attention модуль для фокусировки на важных областях"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.conv1(x)
        attention = self.sigmoid(attention)
        return x * attention

class ChannelAttention(nn.Module):
    """Channel Attention модуль"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class ImprovedCarConditionClassifier(nn.Module):
    def __init__(self, num_classes_clean=2, num_classes_damage=2, dropout_rate=0.3):
        super(ImprovedCarConditionClassifier, self).__init__()
        
        # Используем EfficientNet-B3 как более мощный backbone
        try:
            from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
            self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            print("✅ Используется EfficientNet-B3")
        except Exception as e:
            # Fallback на ResNet50 если EfficientNet недоступен
            print(f"⚠️ EfficientNet недоступен ({e}), используем ResNet50")
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        # Attention механизмы
        self.channel_attention = ChannelAttention(feature_dim)
        self.spatial_attention = SpatialAttention(feature_dim)
        
        # Глобальные признаки
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Сохраняем размерность признаков для адаптации
        self.feature_dim = feature_dim
        self.is_efficientnet = True  # Будет установлено в forward
        
        # Улучшенные классификаторы с более глубокой архитектурой
        self.cleanliness_head = nn.Sequential(
            nn.Linear(feature_dim, 512),  # Для EfficientNet используем feature_dim напрямую
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes_clean)
        )
        
        self.damage_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes_damage)
        )
        
        # Дополнительный классификатор для общей оценки
        self.overall_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 4)  # 4 класса: clean_intact, clean_damaged, dirty_intact, dirty_damaged
        )
        
    def forward(self, x):
        # Извлечение признаков
        features = self.backbone(x)
        
        # Проверяем размерность признаков
        if len(features.shape) == 2:
            # Если признаки уже одномерные (EfficientNet), используем напрямую
            global_features = features
        else:
            # Если признаки многомерные (ResNet), применяем attention
            features = self.channel_attention(features)
            features = self.spatial_attention(features)
            
            # Глобальные признаки
            avg_pool = self.global_pool(features).view(features.size(0), -1)
            max_pool = self.global_max_pool(features).view(features.size(0), -1)
            global_features = torch.cat([avg_pool, max_pool], dim=1)
        
        # Предсказания
        cleanliness_logits = self.cleanliness_head(global_features)
        damage_logits = self.damage_head(global_features)
        overall_logits = self.overall_head(global_features)
        
        return cleanliness_logits, damage_logits, overall_logits

class AdvancedCarAnalyzer:
    def __init__(self, model_path=None, confidence_threshold=0.7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.model = ImprovedCarConditionClassifier()
        self.model.to(self.device)
        
        # Загружаем обученную модель если есть
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.eval()
                print(f"✅ Улучшенная модель загружена: {model_path}")
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
        
        # Улучшенные трансформации
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),  # Увеличили разрешение
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Дополнительные трансформации для аугментации
        self.augment_transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = {
            'cleanliness': ['Чистый', 'Грязный'],
            'damage': ['Целый', 'Битый'],
            'overall': ['Чистый целый', 'Чистый битый', 'Грязный целый', 'Грязный битый']
        }
    
    def preprocess_image(self, image):
        """Улучшенная предобработка изображения"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Детекция автомобиля (простая версия)
        image = self._crop_car_region(image)
        
        return image
    
    def _crop_car_region(self, image):
        """Попытка обрезать изображение до области автомобиля"""
        try:
            # Конвертируем в OpenCV формат
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Простая детекция контуров
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Находим наибольший контур
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Добавляем отступы
                padding = 0.1
                x = max(0, int(x - w * padding))
                y = max(0, int(y - h * padding))
                w = min(cv_image.shape[1] - x, int(w * (1 + 2 * padding)))
                h = min(cv_image.shape[0] - y, int(h * (1 + 2 * padding)))
                
                # Обрезаем изображение
                cropped = cv_image[y:y+h, x:x+w]
                return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            print(f"⚠️ Ошибка обрезки: {e}")
        
        return image
    
    def analyze_image(self, image, use_ensemble=True):
        """Улучшенный анализ изображения с ансамблевыми методами"""
        try:
            # Предобработка
            processed_image = self.preprocess_image(image)
            
            if isinstance(processed_image, torch.Tensor):
                processed_image = processed_image.unsqueeze(0)
            else:
                processed_image = self.transform(processed_image).unsqueeze(0)
            
            processed_image = processed_image.to(self.device)
            
            if self.trained:
                if use_ensemble:
                    return self._ensemble_prediction(processed_image)
                else:
                    return self._single_prediction(processed_image)
            else:
                return self._generate_demo_result()
                
        except Exception as e:
            raise Exception(f"Ошибка анализа: {e}")
    
    def _single_prediction(self, image_tensor):
        """Одиночное предсказание"""
        with torch.no_grad():
            cleanliness_logits, damage_logits, overall_logits = self.model(image_tensor)
            
            # Получаем вероятности
            cleanliness_probs = torch.softmax(cleanliness_logits, dim=1)
            damage_probs = torch.softmax(damage_logits, dim=1)
            overall_probs = torch.softmax(overall_logits, dim=1)
            
            # Предсказания
            cleanliness_pred = torch.argmax(cleanliness_probs, dim=1).item()
            damage_pred = torch.argmax(damage_probs, dim=1).item()
            overall_pred = torch.argmax(overall_probs, dim=1).item()
            
            # Уверенность
            cleanliness_conf = float(cleanliness_probs[0][cleanliness_pred])
            damage_conf = float(damage_probs[0][damage_pred])
            overall_conf = float(overall_probs[0][overall_pred])
            
            # Классы
            cleanliness_class = self.class_names['cleanliness'][cleanliness_pred]
            damage_class = self.class_names['damage'][damage_pred]
            overall_class = self.class_names['overall'][overall_pred]
            
            return self._format_improved_result(
                cleanliness_class, damage_class, overall_class,
                cleanliness_conf, damage_conf, overall_conf, trained=True
            )
    
    def _ensemble_prediction(self, image_tensor):
        """Ансамблевое предсказание с несколькими аугментациями"""
        predictions = []
        
        # Оригинальное изображение
        predictions.append(self._single_prediction(image_tensor))
        
        # Аугментированные версии
        for _ in range(3):
            try:
                # Создаем аугментированную версию
                aug_image = self.augment_transform(self.preprocess_image(image_tensor.squeeze(0).cpu()))
                aug_image = aug_image.unsqueeze(0).to(self.device)
                predictions.append(self._single_prediction(aug_image))
            except:
                continue
        
        # Усредняем предсказания
        return self._average_predictions(predictions)
    
    def _average_predictions(self, predictions):
        """Усреднение нескольких предсказаний"""
        if not predictions:
            return self._generate_demo_result()
        
        # Усредняем уверенность
        avg_clean_conf = np.mean([p['cleanliness']['confidence'] for p in predictions])
        avg_damage_conf = np.mean([p['damage']['confidence'] for p in predictions])
        
        # Берем наиболее частый класс
        clean_classes = [p['cleanliness']['class'] for p in predictions]
        damage_classes = [p['damage']['class'] for p in predictions]
        
        cleanliness_class = max(set(clean_classes), key=clean_classes.count)
        damage_class = max(set(damage_classes), key=damage_classes.count)
        
        return self._format_improved_result(
            cleanliness_class, damage_class, f"{cleanliness_class} {damage_class.lower()}",
            avg_clean_conf, avg_damage_conf, (avg_clean_conf + avg_damage_conf) / 2, trained=True
        )
    
    def _format_improved_result(self, cleanliness_class, damage_class, overall_class, 
                               clean_conf, damage_conf, overall_conf, trained=False):
        """Улучшенное форматирование результата"""
        
        # Определяем общее состояние с учетом уверенности
        if clean_conf > self.confidence_threshold and damage_conf > self.confidence_threshold:
            if cleanliness_class == 'Чистый' and damage_class == 'Целый':
                overall = 'Отличное'
                quality_score = 95
            elif cleanliness_class == 'Грязный' and damage_class == 'Целый':
                overall = 'Хорошее (требует мойки)'
                quality_score = 75
            elif cleanliness_class == 'Чистый' and damage_class == 'Битый':
                overall = 'Требует ремонта'
                quality_score = 60
            else:
                overall = 'Плохое (требует ремонта и мойки)'
                quality_score = 40
        else:
            overall = 'Неопределенное (низкая уверенность)'
            quality_score = 50
        
        # Улучшенные рекомендации
        recommendations = []
        
        # Рекомендации по чистоте
        if cleanliness_class == 'Грязный':
            if clean_conf > 0.8:
                recommendations.append('🚿 Автомобиль определенно нуждается в мойке')
                recommendations.append('💡 Рекомендуется полная мойка кузова и салона')
            else:
                recommendations.append('🧽 Рекомендуется помыть автомобиль')
            recommendations.append('⭐ Чистый автомобиль повышает рейтинг водителя')
            recommendations.append('📸 Сделайте фото после мойки для подтверждения')
        
        # Рекомендации по повреждениям
        if damage_class == 'Битый':
            if damage_conf > 0.8:
                recommendations.append('⚠️ Обнаружены серьезные повреждения')
                recommendations.append('🔧 Требуется немедленный техосмотр и ремонт')
                recommendations.append('📞 Обратитесь в техподдержку inDrive')
            else:
                recommendations.append('🔍 Возможны повреждения - рекомендуется осмотр')
                recommendations.append('📷 Сделайте дополнительные фото для оценки')
            recommendations.append('🚫 Не рекомендуется принимать заказы до устранения')
        
        # Общие рекомендации
        if not recommendations:
            recommendations.append('✅ Автомобиль в отличном состоянии!')
            recommendations.append('🌟 Поддерживайте такое состояние для высокого рейтинга')
            recommendations.append('�� Продолжайте качественную работу')
        
        # Добавляем рекомендации по качеству
        if quality_score < 70:
            recommendations.append('📊 Рекомендуется улучшить общее состояние автомобиля')
        
        return {
            'cleanliness': {
                'class': cleanliness_class,
                'confidence': clean_conf,
                'status': 'high' if clean_conf > 0.8 else 'medium' if clean_conf > 0.6 else 'low'
            },
            'damage': {
                'class': damage_class,
                'confidence': damage_conf,
                'status': 'high' if damage_conf > 0.8 else 'medium' if damage_conf > 0.6 else 'low'
            },
            'overall_condition': overall,
            'quality_score': quality_score,
            'overall_class': overall_class,
            'overall_confidence': overall_conf,
            'recommendations': recommendations,
            'model_trained': trained,
            'analysis_quality': 'high' if min(clean_conf, damage_conf) > 0.7 else 'medium'
        }
    
    def _generate_demo_result(self):
        """Генерация улучшенных демо-результатов"""
        scenarios = [
            {'clean': 0, 'damage': 0, 'clean_conf': 0.94, 'damage_conf': 0.91},
            {'clean': 1, 'damage': 0, 'clean_conf': 0.89, 'damage_conf': 0.95},
            {'clean': 0, 'damage': 1, 'clean_conf': 0.78, 'damage_conf': 0.85},
            {'clean': 1, 'damage': 1, 'clean_conf': 0.92, 'damage_conf': 0.81},
        ]
        
        scenario = random.choice(scenarios)
        
        # Добавляем реалистичную случайность
        clean_conf = max(0.6, min(0.99, scenario['clean_conf'] + random.uniform(-0.08, 0.08)))
        damage_conf = max(0.6, min(0.99, scenario['damage_conf'] + random.uniform(-0.08, 0.08)))
        
        cleanliness_class = self.class_names['cleanliness'][scenario['clean']]
        damage_class = self.class_names['damage'][scenario['damage']]
        overall_class = f"{cleanliness_class} {damage_class.lower()}"
        
        return self._format_improved_result(
            cleanliness_class, damage_class, overall_class,
            clean_conf, damage_conf, (clean_conf + damage_conf) / 2, trained=False
        )

# Для обратной совместимости
CarConditionClassifier = ImprovedCarConditionClassifier
CarAnalyzer = AdvancedCarAnalyzer
