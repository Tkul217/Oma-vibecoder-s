from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from PIL import Image
import io
import os
import time
import logging
from car_model import CarAnalyzer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Инициализируем анализатор с обученной моделью
try:
    # Сначала пытаемся загрузить обученную модель
    model_path = 'best_model.pth' if os.path.exists('best_model.pth') else None
    analyzer = CarAnalyzer(model_path=model_path)
    logger.info("✅ ML модель загружена успешно")
except Exception as e:
    analyzer = None
    logger.error(f"❌ Ошибка загрузки модели: {e}")

# HTML шаблон для веб-интерфейса
WEB_INTERFACE = '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚗 inDrive Car Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .upload-area { border: 2px dashed #ccc; border-radius: 10px; padding: 40px; text-align: center; margin: 20px 0; }
        .upload-area:hover { border-color: #007bff; }
        .result { margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 10px; }
        .success { background: #d4edda; }
        .warning { background: #fff3cd; }
        .danger { background: #f8d7da; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .hidden { display: none; }
        img { max-width: 100%; height: auto; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>🚗 inDrive Car Analysis API</h1>
    <p>Загрузите фотографию автомобиля для анализа состояния</p>
    
    <div class="upload-area" onclick="document.getElementById('fileInput').click()">
        <p>📸 Нажмите здесь или перетащите изображение</p>
        <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="uploadImage()">
    </div>
    
    <div id="loading" class="hidden">⏳ Анализ изображения...</div>
    <div id="preview"></div>
    <div id="results"></div>
    
    <h3>📡 API Endpoints:</h3>
    <ul>
        <li><a href="/health">GET /health</a> - Проверка состояния</li>
        <li>POST /predict - Анализ изображения</li>
        <li><a href="/info">GET /info</a> - Информация об API</li>
    </ul>

    <script>
        async function uploadImage() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) return;
            
            // Показать превью
            const preview = document.getElementById('preview');
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            preview.innerHTML = '<h3>📷 Загруженное изображение:</h3>';
            preview.appendChild(img);
            
            // Показать загрузку
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('results').innerHTML = '';
            
            // Отправить на анализ
            const formData = new FormData();
            formData.append('image', file);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                displayResults(data);
                
            } catch (error) {
                displayResults({success: false, error: 'Ошибка соединения с сервером'});
            }
            
            document.getElementById('loading').classList.add('hidden');
        }
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            
            if (data.success) {
                resultsDiv.innerHTML = `
                    <div class="result success">
                        <h3>✅ Результаты анализа</h3>
                        <p><strong>🧽 Чистота:</strong> ${data.cleanliness.class} (${Math.round(data.cleanliness.confidence * 100)}%)</p>
                        <p><strong>🔧 Повреждения:</strong> ${data.damage.class} (${Math.round(data.damage.confidence * 100)}%)</p>
                        <p><strong>📊 Общее состояние:</strong> ${data.overall_condition}</p>
                        <p><strong>🤖 Модель:</strong> ${data.model_trained ? 'Обученная' : 'Демо режим'}</p>
                        <h4>💡 Рекомендации:</h4>
                        <ul>${data.recommendations.map(r => '<li>' + r + '</li>').join('')}</ul>
                        <small>⏱️ Время обработки: ${data.metadata?.processing_time_ms || 'N/A'} мс</small>
                    </div>
                `;
            } else {
                resultsDiv.innerHTML = `
                    <div class="result danger">
                        <h3>❌ Ошибка</h3>
                        <p>${data.error}</p>
                    </div>
                `;
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    """Главная страница с веб-интерфейсом"""
    return render_template_string(WEB_INTERFACE)

@app.route('/health')
def health():
    """Проверка состояния API"""
    return jsonify({
        'status': 'healthy',
        'service': 'inDrive Car Analysis API',
        'version': '1.0.0',
        'model_loaded': analyzer is not None,
        'model_trained': analyzer.trained if analyzer else False,
        'endpoints': ['/', '/health', '/predict', '/info']
    })

@app.route('/info')
def info():
    """Информация об API"""
    return jsonify({
        'service': 'inDrive Car Analysis API',
        'description': 'Анализ состояния автомобилей по фотографиям',
        'version': '1.0.0',
        'model_status': 'loaded' if analyzer else 'error',
        'model_trained': analyzer.trained if analyzer else False,
        'capabilities': [
            'Определение чистоты (чистый/грязный)',
            'Определение повреждений (целый/битый)',
            'Общая оценка состояния',
            'Рекомендации по улучшению'
        ],
        'usage': {
            'method': 'POST',
            'endpoint': '/predict',
            'content_type': 'multipart/form-data',
            'field_name': 'image',
            'max_size': '10MB'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Анализ изображения автомобиля"""
    start_time = time.time()
    
    try:
        # Проверяем наличие модели
        if analyzer is None:
            return jsonify({
                'success': False,
                'error': 'ML модель не загружена'
            }), 500
        
        # Проверяем файл
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Изображение не найдено в запросе'
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Файл не выбран'
            }), 400
        
        # Проверяем размер
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            return jsonify({
                'success': False,
                'error': 'Файл слишком большой (максимум 10MB)'
            }), 400
        
        # Загружаем изображение
        try:
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            logger.info(f"Анализ изображения: {file.filename}, размер: {image.size}")
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Не удалось обработать изображение: {str(e)}'
            }), 400
        
        # Анализируем
        result = analyzer.analyze_image(image)
        
        # Добавляем метаданные
        processing_time = round((time.time() - start_time) * 1000, 2)
        result.update({
            'success': True,
            'filename': file.filename,
            'file_size_kb': round(file_size / 1024, 2),
            'metadata': {
                'processing_time_ms': processing_time,
                'image_size': list(image.size),
                'model_version': 'trained-1.0' if analyzer.trained else 'demo-1.0'
            }
        })
        
        logger.info(f"Анализ завершен: {result['overall_condition']}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Ошибка анализа: {e}")
        return jsonify({
            'success': False,
            'error': f'Ошибка сервера: {str(e)}'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5010))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    print("=" * 60)
    print("🚗 inDrive Car Analysis API")
    print("=" * 60)
    print(f"🌐 Веб-интерфейс: http://localhost:{port}")
    print(f"📡 API: http://localhost:{port}/predict") 
    print(f"❤️  Health: http://localhost:{port}/health")
    print(f"🔧 Debug: {debug}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=debug)