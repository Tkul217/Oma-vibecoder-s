from flask import Flask, request, jsonify, render_template_string
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
    logger.error(f"❌ Ошибка загрузки модели: {e}")

# Улучшенный HTML шаблон
WEB_INTERFACE = '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚗 inDrive Advanced Car Analysis</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { 
            max-width: 900px; 
            margin: 0 auto; 
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .content { padding: 40px; }
        .upload-area { 
            border: 3px dashed #667eea; 
            border-radius: 15px; 
            padding: 60px 40px; 
            text-align: center; 
            margin: 30px 0;
            transition: all 0.3s ease;
            cursor: pointer;
            background: #f8f9ff;
        }
        .upload-area:hover { 
            border-color: #764ba2; 
            background: #f0f2ff;
            transform: translateY(-2px);
        }
        .upload-area.dragover {
            border-color: #4CAF50;
            background: #e8f5e8;
        }
        .result { 
            margin: 30px 0; 
            padding: 25px; 
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .success { background: linear-gradient(135deg, #d4edda, #c3e6cb); }
        .warning { background: linear-gradient(135deg, #fff3cd, #ffeaa7); }
        .danger { background: linear-gradient(135deg, #f8d7da, #f5c6cb); }
        .info { background: linear-gradient(135deg, #d1ecf1, #bee5eb); }
        button { 
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white; 
            padding: 15px 30px; 
            border: none; 
            border-radius: 25px; 
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        button:hover { 
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        .hidden { display: none; }
        img { 
            max-width: 100%; 
            height: auto; 
            margin: 15px 0; 
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        .confidence-bar {
            background: #e9ecef;
            border-radius: 10px;
            height: 20px;
            margin: 10px 0;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        .high-confidence { background: linear-gradient(90deg, #28a745, #20c997); }
        .medium-confidence { background: linear-gradient(90deg, #ffc107, #fd7e14); }
        .low-confidence { background: linear-gradient(90deg, #dc3545, #e83e8c); }
        .quality-score {
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }
        .score-excellent { color: #28a745; }
        .score-good { color: #ffc107; }
        .score-poor { color: #dc3545; }
        .recommendations {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .recommendations ul {
            list-style: none;
            padding: 0;
        }
        .recommendations li {
            padding: 8px 0;
            border-bottom: 1px solid #dee2e6;
        }
        .recommendations li:last-child {
            border-bottom: none;
        }
        .api-info {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .api-info h3 {
            color: #667eea;
            margin-bottom: 15px;
        }
        .api-info ul {
            list-style: none;
            padding: 0;
        }
        .api-info li {
            padding: 5px 0;
            border-bottom: 1px solid #dee2e6;
        }
        .api-info a {
            color: #667eea;
            text-decoration: none;
        }
        .api-info a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚗 inDrive Advanced Car Analysis</h1>
            <p>Продвинутый анализ состояния автомобилей с использованием ИИ</p>
        </div>
        
        <div class="content">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()" 
                 ondrop="handleDrop(event)" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)">
                <div style="font-size: 3em; margin-bottom: 20px;">📸</div>
                <h3>Загрузите фотографию автомобиля</h3>
                <p>Нажмите здесь или перетащите изображение для анализа</p>
                <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="uploadImage()">
            </div>
            
            <div id="loading" class="hidden">
                <div style="text-align: center; padding: 40px;">
                    <div style="font-size: 2em; margin-bottom: 20px;">⏳</div>
                    <h3>Анализ изображения...</h3>
                    <p>Пожалуйста, подождите</p>
                </div>
            </div>
            
            <div id="preview"></div>
            <div id="results"></div>
            
            <div class="api-info">
                <h3>📡 API Endpoints:</h3>
                <ul>
                    <li><a href="/health">GET /health</a> - Проверка состояния</li>
                    <li>POST /predict - Анализ изображения</li>
                    <li><a href="/info">GET /info</a> - Информация об API</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        function handleDragOver(e) {
            e.preventDefault();
            e.currentTarget.classList.add('dragover');
        }
        
        function handleDragLeave(e) {
            e.currentTarget.classList.remove('dragover');
        }
        
        function handleDrop(e) {
            e.preventDefault();
            e.currentTarget.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('fileInput').files = files;
                uploadImage();
            }
        }
        
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
                const cleanliness = data.cleanliness;
                const damage = data.damage;
                const qualityScore = data.quality_score || 0;
                
                // Определяем класс для качества
                let scoreClass = 'score-poor';
                if (qualityScore >= 80) scoreClass = 'score-excellent';
                else if (qualityScore >= 60) scoreClass = 'score-good';
                
                // Определяем класс для уверенности
                function getConfidenceClass(conf) {
                    if (conf >= 0.8) return 'high-confidence';
                    if (conf >= 0.6) return 'medium-confidence';
                    return 'low-confidence';
                }
                
                resultsDiv.innerHTML = `
                    <div class="result success">
                        <h3>✅ Результаты анализа</h3>
                        
                        <div class="quality-score ${scoreClass}">
                            ${qualityScore}%
                        </div>
                        
                        <div class="metric-card">
                            <h4>🧽 Чистота: ${cleanliness.class}</h4>
                            <div class="confidence-bar">
                                <div class="confidence-fill ${getConfidenceClass(cleanliness.confidence)}" 
                                     style="width: ${cleanliness.confidence * 100}%"></div>
                            </div>
                            <p>Уверенность: ${Math.round(cleanliness.confidence * 100)}% (${cleanliness.status})</p>
                        </div>
                        
                        <div class="metric-card">
                            <h4>🔧 Повреждения: ${damage.class}</h4>
                            <div class="confidence-bar">
                                <div class="confidence-fill ${getConfidenceClass(damage.confidence)}" 
                                     style="width: ${damage.confidence * 100}%"></div>
                            </div>
                            <p>Уверенность: ${Math.round(damage.confidence * 100)}% (${damage.status})</p>
                        </div>
                        
                        <div class="metric-card">
                            <h4>📊 Общее состояние: ${data.overall_condition}</h4>
                            <p>Класс: ${data.overall_class || 'N/A'}</p>
                            <p>Качество анализа: ${data.analysis_quality || 'N/A'}</p>
                        </div>
                        
                        <div class="recommendations">
                            <h4>💡 Рекомендации:</h4>
                            <ul>${data.recommendations.map(r => '<li>' + r + '</li>').join('')}</ul>
                        </div>
                        
                        <div style="text-align: center; margin-top: 20px; color: #666;">
                            <small>⏱️ Время обработки: ${data.metadata?.processing_time_ms || 'N/A'} мс</small>
                            <br>
                            <small>🤖 Модель: ${data.model_trained ? 'Обученная (улучшенная)' : 'Демо режим'}</small>
                        </div>
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
    """Главная страница с улучшенным веб-интерфейсом"""
    return render_template_string(WEB_INTERFACE)

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
        'endpoints': ['/', '/health', '/predict', '/info']
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
            'field_name': 'image',
            'max_size': '10MB',
            'supported_formats': ['JPEG', 'PNG', 'BMP', 'WEBP']
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Улучшенный анализ изображения автомобиля"""
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
        
        # Анализируем с ансамблевыми методами
        result = analyzer.analyze_image(image, use_ensemble=True)
        
        # Добавляем метаданные
        processing_time = round((time.time() - start_time) * 1000, 2)
        result.update({
            'success': True,
            'filename': file.filename,
            'file_size_kb': round(file_size / 1024, 2),
            'metadata': {
                'processing_time_ms': processing_time,
                'image_size': list(image.size),
                'model_version': 'improved-2.0' if analyzer.trained else 'demo-2.0',
                'ensemble_used': True,
                'confidence_threshold': analyzer.confidence_threshold if hasattr(analyzer, 'confidence_threshold') else 0.5
            }
        })
        
        logger.info(f"Анализ завершен: {result['overall_condition']}, качество: {result.get('quality_score', 'N/A')}")
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
    
    print("=" * 70)
    print("🚗 inDrive Advanced Car Analysis API v2.0")
    print("=" * 70)
    print(f"🌐 Веб-интерфейс: http://localhost:{port}")
    print(f"📡 API: http://localhost:{port}/predict") 
    print(f"❤️  Health: http://localhost:{port}/health")
    print(f"🔧 Debug: {debug}")
    print(f"🤖 Модель: {'Улучшенная' if analyzer and hasattr(analyzer, 'confidence_threshold') else 'Базовая'}")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=port, debug=debug)
