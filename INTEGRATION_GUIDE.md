# Интеграция Python ML модели с приложением

## 🎯 Варианты интеграции

### Вариант 1: Python микросервис (Рекомендуемый)

Создайте отдельный Python API сервер:

```python
# ml_service/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

# Загрузка вашей модели
class CarAnalyzer:
    def __init__(self):
        # Загрузите ваши обученные модели
        self.cleanliness_model = torch.load('models/cleanliness_model.pth')
        self.condition_model = torch.load('models/condition_model.pth')
        self.cleanliness_model.eval()
        self.condition_model.eval()
    
    def analyze(self, image):
        # Ваша логика анализа
        # Возвращает результат анализа
        pass

analyzer = CarAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze_car():
    try:
        # Получение изображения
        file = request.files['image']
        image = Image.open(file.stream)
        
        # Анализ с помощью ML модели
        result = analyzer.analyze(image)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

```bash
# ml_service/requirements.txt
flask==2.3.3
flask-cors==4.0.0
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.3.0
numpy>=1.21.0
```

### Вариант 2: Прямой вызов Python скрипта

Обновите Node.js сервер для вызова Python:

```javascript
// server/server.js - добавить в функцию analyzeCarImage
const { spawn } = require('child_process');
const path = require('path');

function analyzeCarImage(imagePath) {
  return new Promise((resolve, reject) => {
    const pythonScript = path.join(__dirname, '../ml/analyze.py');
    const python = spawn('python', [pythonScript, imagePath]);
    
    let result = '';
    let error = '';
    
    python.stdout.on('data', (data) => {
      result += data.toString();
    });
    
    python.stderr.on('data', (data) => {
      error += data.toString();
    });
    
    python.on('close', (code) => {
      if (code === 0) {
        try {
          const analysisResult = JSON.parse(result);
          resolve(analysisResult);
        } catch (e) {
          reject(new Error('Invalid JSON response from ML model'));
        }
      } else {
        reject(new Error(`ML analysis failed: ${error}`));
      }
    });
  });
}
```

## 🚀 Пошаговая интеграция

### Шаг 1: Подготовка ML модели

Ваш друг должен создать скрипт `ml/analyze.py`:

```python
#!/usr/bin/env python3
import sys
import json
import torch
from PIL import Image
import torchvision.transforms as transforms

def load_models():
    """Загрузка обученных моделей"""
    cleanliness_model = torch.load('models/cleanliness_model.pth', map_location='cpu')
    condition_model = torch.load('models/condition_model.pth', map_location='cpu')
    
    cleanliness_model.eval()
    condition_model.eval()
    
    return cleanliness_model, condition_model

def preprocess_image(image_path):
    """Предобработка изображения"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def analyze_image(image_path):
    """Анализ изображения"""
    try:
        # Загрузка моделей
        cleanliness_model, condition_model = load_models()
        
        # Предобработка
        image_tensor = preprocess_image(image_path)
        
        # Предсказания
        with torch.no_grad():
            # Анализ чистоты
            clean_outputs = cleanliness_model(image_tensor)
            clean_probs = torch.nn.functional.softmax(clean_outputs, dim=1)
            clean_confidence = float(clean_probs.max())
            cleanliness = 'clean' if clean_outputs.argmax() == 0 else 'dirty'
            
            # Анализ повреждений
            condition_outputs = condition_model(image_tensor)
            condition_probs = torch.nn.functional.softmax(condition_outputs, dim=1)
            condition_confidence = float(condition_probs.max())
            condition = 'intact' if condition_outputs.argmax() == 0 else 'damaged'
        
        return {
            'cleanliness': cleanliness,
            'cleanlinessConfidence': clean_confidence,
            'condition': condition,
            'conditionConfidence': condition_confidence,
            'processingTime': 1200  # Примерное время
        }
        
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(json.dumps({'error': 'Usage: python analyze.py <image_path>'}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = analyze_image(image_path)
    print(json.dumps(result))
```

### Шаг 2: Структура проекта

```
project/
├── src/                    # React frontend
├── server/                 # Node.js API
├── ml/                     # Python ML код
│   ├── analyze.py         # Скрипт анализа
│   ├── models/            # Обученные модели
│   │   ├── cleanliness_model.pth
│   │   └── condition_model.pth
│   └── requirements.txt
└── ml_service/            # Опциональный Python API
    ├── app.py
    └── requirements.txt
```

### Шаг 3: Настройка окружения

```bash
# Установка Python зависимостей
cd ml
pip install -r requirements.txt

# Или для микросервиса
cd ml_service
pip install -r requirements.txt
```

### Шаг 4: Обновление Node.js сервера

Если используете микросервис, обновите URL в server.js:

```javascript
// Вместо mock функции
async function analyzeCarImage(imagePath) {
  try {
    const formData = new FormData();
    const fileStream = fs.createReadStream(imagePath);
    formData.append('image', fileStream);
    
    const response = await fetch('http://localhost:5000/analyze', {
      method: 'POST',
      body: formData
    });
    
    return await response.json();
  } catch (error) {
    throw new Error(`ML service error: ${error.message}`);
  }
}
```

## 🔧 Конфигурация для продакшена

### Docker Compose для всех сервисов

```yaml
# docker-compose.yml
version: '3.8'
services:
  frontend:
    build: .
    ports:
      - "3000:3000"
    depends_on:
      - api
      - ml-service

  api:
    build: ./server
    ports:
      - "3001:3001"
    depends_on:
      - ml-service

  ml-service:
    build: ./ml_service
    ports:
      - "5000:5000"
    volumes:
      - ./ml/models:/app/models
```

### Dockerfile для ML сервиса

```dockerfile
# ml_service/Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

## 🧪 Тестирование интеграции

```bash
# Тест Python скрипта
python ml/analyze.py path/to/test/image.jpg

# Тест микросервиса
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/analyze

# Тест полной системы
# Загрузите изображение через веб-интерфейс
```

## 📋 Чеклист интеграции

- [ ] ML модели обучены и сохранены в формате PyTorch
- [ ] Создан скрипт анализа `analyze.py`
- [ ] Установлены Python зависимости
- [ ] Обновлен Node.js сервер для вызова Python
- [ ] Протестирована работа на тестовых изображениях
- [ ] Настроена обработка ошибок
- [ ] Добавлено логирование

## 🚨 Возможные проблемы

1. **Путь к Python**: Убедитесь что `python` доступен в PATH
2. **Зависимости**: Все библиотеки установлены в правильном окружении
3. **Размер моделей**: Большие модели могут долго загружаться
4. **Память**: PyTorch модели требуют достаточно RAM
5. **CORS**: Настройте правильные заголовки для микросервиса

## 💡 Рекомендации

- Используйте виртуальное окружение Python
- Кешируйте загруженные модели в памяти
- Добавьте мониторинг производительности
- Реализуйте graceful shutdown для сервисов
- Используйте GPU если доступно (`torch.cuda.is_available()`)