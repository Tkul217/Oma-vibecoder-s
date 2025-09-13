# Car Condition Analyzer

🚗 Система анализа состояния автомобилей по фотографиям с использованием машинного обучения.

## 📥 Быстрый старт

### 1. Скачайте проект
```bash
git clone https://github.com/your-username/car-condition-analyzer.git
cd car-condition-analyzer
```

### 2. Установите все зависимости одной командой
```bash
npm run setup
```

### 3. Запустите проект
```bash
npm run start:dev
```

Откройте http://localhost:5173 - готово! 🎉

## 🚀 Возможности

- **Анализ чистоты**: Определение грязного или чистого автомобиля
- **Оценка повреждений**: Выявление целого или битого состояния
- **Веб-интерфейс**: Удобная загрузка фотографий с drag-and-drop
- **API**: RESTful API для интеграции с другими системами
- **Модульная архитектура**: Готова для интеграции с реальными ML-моделями

## 🏗️ Архитектура

```
├── src/                 # React фронтенд
├── server/             # Express.js API сервер
├── ml/                 # ML модели и обучение
│   ├── model_integration.py   # Интеграция с моделями
│   ├── train_model.py        # Обучение моделей
│   └── requirements.txt      # Python зависимости
└── README.md
```

## 🛠️ Технологии

### Frontend
- **React 18** с TypeScript
- **Tailwind CSS** для стилизации
- **Lucide React** для иконок
- **Vite** для сборки

### Backend
- **Node.js** с Express
- **Multer** для загрузки файлов
- Proxy к ML сервису

### ML Stack
- **PyTorch** для глубокого обучения
- **torchvision** для обработки изображений
- **PIL/Pillow** для работы с изображениями
- **ResNet50** как backbone модель

## 🚀 Быстрый запуск

### 1. Установка зависимостей
```bash
# Установка frontend зависимостей
npm install

# Установка backend зависимостей
cd server && npm install
```

### 2. Запуск разработки
```bash
# Терминал 1: Запуск API сервера
cd server && npm start

# Терминал 2: Запуск frontend
npm run dev
```

### 3. Настройка ML окружения (опционально)
```bash
cd ml
pip install -r requirements.txt
```

## 🧠 Интеграция с ML моделями

### Текущее состояние
Система работает с заглушками ML моделей для демонстрации архитектуры.

### Интеграция реальных моделей

#### 1. Подготовка данных
Создайте структуру данных:
```
data/
  train/
    clean_intact/
    clean_damaged/
    dirty_intact/
    dirty_damaged/
  val/
    clean_intact/
    clean_damaged/
    dirty_intact/
    dirty_damaged/
```

#### 2. Создание аннотаций
```json
[
  {
    "filename": "car_001.jpg",
    "cleanliness": "clean",
    "condition": "intact"
  },
  {
    "filename": "car_002.jpg",
    "cleanliness": "dirty",
    "condition": "damaged"
  }
]
```

#### 3. Обучение моделей
```bash
cd ml
python train_model.py
```

#### 4. Замена заглушек
В файле `server/server.js` замените функцию `analyzeCarImage`:

```javascript
// Вместо mock функции
const { spawn } = require('child_process');

function analyzeCarImage(imagePath) {
  return new Promise((resolve, reject) => {
    const python = spawn('python', ['../ml/model_integration.py', imagePath]);
    
    let result = '';
    python.stdout.on('data', (data) => {
      result += data.toString();
    });
    
    python.on('close', (code) => {
      if (code === 0) {
        resolve(JSON.parse(result));
      } else {
        reject(new Error('ML analysis failed'));
      }
    });
  });
}
```

#### 5. Python микросервис (альтернативно)
Создайте отдельный Python API:
```bash
cd ml
pip install flask flask-cors
python -c "
from flask import Flask, request, jsonify
from model_integration import CarConditionAnalyzer

app = Flask(__name__)
analyzer = CarConditionAnalyzer('models/')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']
    # Обработка и анализ
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)
"
```

## 📋 API Документация

### POST /api/analyze
Анализ изображения автомобиля.

**Request:**
```
Content-Type: multipart/form-data
Body: image file
```

**Response:**
```json
{
  "cleanliness": "clean",
  "cleanlinessConfidence": 0.89,
  "condition": "intact", 
  "conditionConfidence": 0.92,
  "processingTime": 1243
}
```

### GET /api/health
Проверка состояния API.

**Response:**
```json
{
  "status": "OK",
  "message": "Car Condition Analyzer API is running"
}
```

## 🔧 Конфигурация

### Настройки сервера (server/server.js)
- `PORT`: Порт API сервера (по умолчанию 3001)
- `fileSize`: Максимальный размер файла (10MB)
- `allowedTypes`: Поддерживаемые форматы изображений

### Настройки ML (ml/model_integration.py)
- `device`: CUDA или CPU
- `image_size`: Размер входного изображения (224x224)
- `confidence_threshold`: Порог уверенности модели

## 🐛 Решение проблем

### Распространенные ошибки

1. **File too large**
   - Увеличьте лимит в `multer` конфигурации
   - Оптимизируйте изображения перед загрузкой

2. **ML analysis failed**
   - Проверьте установку Python зависимостей
   - Убедитесь что модели доступны по указанному пути

3. **CORS errors**
   - Настройте правильные заголовки в API
   - Проверьте proxy конфигурацию в Vite

## 🚀 Деплой

### Frontend (Vercel/Netlify)
```bash
npm run build
# Загрузите папку dist
```

### Backend (Railway/Heroku)
```bash
# Создайте Dockerfile или используйте package.json
# Настройте переменные окружения
```

### ML сервис (Docker)
```dockerfile
FROM python:3.9
COPY ml/ /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD python model_integration.py
```

## 📈 Дальнейшее развитие

- [ ] Добавление batch обработки изображений
- [ ] Интеграция с облачными ML сервисами (AWS SageMaker)
- [ ] Добавление детекции типа автомобиля
- [ ] Расширенная аналитика и отчеты
- [ ] Мобильное приложение
- [ ] API rate limiting и аутентификация

## 📄 Лицензия

MIT License - свободно используйте в своих проектах.