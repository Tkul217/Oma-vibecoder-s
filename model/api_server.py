from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import os
import time
import logging
from improved_car_model import AdvancedCarAnalyzer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Инициализируем улучшенный анализатор
try:
    # Сначала пытаемся загрузить улучшенную модель
    model_path = 'best_improved_model.pth' if os.path.exists('best_improved_model.pth') else 'best_model.pth'
    analyzer = AdvancedCarAnalyzer(model_path=model_path, confidence_threshold=0.7)
    logger.info("✅ Улучшенная ML модель загружена успешно")
except Exception as e:
    analyzer = None
    logger.error("❌ Ошибка загрузки модели: %s", e)

@app.route('/health')
def health():
    """Проверка состояния API"""
    return jsonify({
        'status': 'healthy',
        'service': 'inDrive Advanced Car Analysis API',
        'version': '2.0.0',
        'model_loaded': analyzer is not None,
        'model_trained': analyzer.trained if analyzer else False,
        'model_type': 'improved' if analyzer and hasattr(analyzer, 'confidence_threshold') else 'basic',
        'endpoints': ['/health', '/predict', '/info']
    })

@app.route('/info')
def info():
    """Информация об API"""
    return jsonify({
        'service': 'inDrive Advanced Car Analysis API',
        'description': 'Продвинутый анализ состояния автомобилей с использованием ИИ',
        'version': '2.0.0',
        'model_status': 'loaded' if analyzer else 'error',
        'model_trained': analyzer.trained if analyzer else False,
        'model_type': 'improved' if analyzer and hasattr(analyzer, 'confidence_threshold') else 'basic',
        'capabilities': [
            'Определение чистоты (чистый/грязный) с высокой точностью',
            'Определение повреждений (целый/битый) с детальной оценкой',
            'Общая оценка состояния с качественным скором',
            'Умные рекомендации по улучшению',
            'Ансамблевые методы для повышения точности',
            'Детекция и обрезка области автомобиля',
            'Оценка качества анализа'
        ],
        'improvements': [
            'EfficientNet-B3 backbone для лучшего извлечения признаков',
            'Attention механизмы (Spatial + Channel)',
            'Focal Loss для работы с несбалансированными данными',
            'Продвинутая аугментация данных',
            'Early stopping и learning rate scheduling',
            'Ансамблевые предсказания',
            'Улучшенная предобработка изображений'
        ],
        'usage': {
            'method': 'POST',
            'endpoint': '/predict',
            'content_type': 'multipart/form-data',
            'parameter': 'image (файл изображения)',
            'response_format': 'JSON'
        },
        'team': 'oma vibecoders'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Анализ изображения автомобиля"""
    start_time = time.time()
    
    if not analyzer:
        return jsonify({
            'error': 'Модель не загружена',
            'status': 'error'
        }), 500
    
    if 'image' not in request.files:
        return jsonify({
            'error': 'Файл изображения не найден',
            'status': 'error'
        }), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({
            'error': 'Файл не выбран',
            'status': 'error'
        }), 400
    
    try:
        # Читаем изображение
        image = Image.open(io.BytesIO(file.read()))
        
        # Конвертируем в RGB если нужно
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info("Анализируем изображение: %s, размер: %s", file.filename, image.size)
        
        # Анализируем изображение
        result = analyzer.analyze_image(image)
        
        processing_time = (time.time() - start_time) * 1000  # в миллисекундах
        
        # Формируем ответ
        response = {
            'status': 'success',
            'cleanliness': {
                'class': result['cleanliness']['class'],
                'confidence': float(result['cleanliness']['confidence']),
            },
            'damage': {
                'class': result['damage']['class'],
                'confidence': float(result['damage']['confidence']),
            },
            'quality_score': result.get('quality_score', None),
            'overall_condition': result.get('overall_condition', None),
            'recommendations': result.get('recommendations', []),
            'metadata': {
                'processing_time_ms': round(processing_time, 2),
                'model_type': 'improved',
                'image_size': image.size,
                'filename': file.filename
            }
        }
        
        logger.info("Анализ завершен за %.2fмс", processing_time)
        return jsonify(response)
        
    except Exception as e:
        logger.error("Ошибка анализа: %s", e)
        return jsonify({
            'error': f'Ошибка обработки изображения: {str(e)}',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    print("🚗 Запуск inDrive Car Analysis API (только API)")
    print("=" * 50)
    print("📡 API доступен по адресу: http://localhost:5010")
    print("🔗 Endpoints:")
    print("   - POST /predict - Анализ изображения")
    print("   - GET /health - Проверка состояния")
    print("   - GET /info - Информация об API")
    print("=" * 50)
    print("⚛️  Для веб-интерфейса используйте React фронтенд")
    print("   Запустите: npm run dev")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5010, debug=False)
