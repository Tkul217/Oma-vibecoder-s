# 📥 Как скачать и запустить проект локально

## 🎯 Простой способ - скачать ZIP

1. **Нажмите кнопку "Download" в правом верхнем углу Bolt**
2. **Выберите "Download as ZIP"**
3. **Распакуйте архив** на своем компьютере
4. **Готово!** У вас есть вся папка с проектом

## 🔧 Настройка локально

### 1. Установите зависимости для фронтенда
```bash
npm install
```

### 2. Установите зависимости для бэкенда
```bash
cd server
npm install
cd ..
```

### 3. Установите Python зависимости для ML
```bash
cd ml
pip install -r requirements.txt
cd ..
```

## 🚀 Запуск проекта

### Вариант 1: Запуск всего сразу
```bash
# Терминал 1: Запуск бэкенда
cd server
npm start

# Терминал 2: Запуск фронтенда  
npm run dev
```

### Вариант 2: Используйте package.json скрипты
Добавлю удобные команды для запуска всего проекта одной командой.

## 📁 Структура проекта
```
car-condition-analyzer/
├── src/                    # React фронтенд
├── server/                 # Node.js API
├── ml/                     # Python ML код
│   ├── train_unified_model.py
│   ├── analyze.py
│   ├── requirements.txt
│   └── models/            # Сюда сохраняются модели
├── data/                  # Сюда кладете фото для обучения
│   ├── clean_intact/
│   ├── clean_damaged/
│   ├── dirty_intact/
│   └── dirty_damaged/
├── package.json
└── README.md
```

## 🔄 Загрузка в GitHub

1. **Создайте репозиторий** на GitHub
2. **Инициализируйте git** в папке проекта:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/ваш-username/car-analyzer.git
git push -u origin main
```

## 🧠 Обучение ML модели локально

1. **Подготовьте данные** - разложите фото по папкам в `data/`
2. **Запустите обучение**:
```bash
cd ml
python train_unified_model.py
```
3. **Дождитесь завершения** - модель сохранится в `ml/models/`

## 🌐 Деплой

После локальной разработки можете задеплоить:
- **Фронтенд**: Vercel, Netlify
- **Бэкенд**: Railway, Heroku
- **ML**: Docker контейнер

## ❓ Возможные проблемы

**Node.js не установлен?**
- Скачайте с nodejs.org

**Python не установлен?**  
- Скачайте с python.org

**Ошибки с зависимостями?**
- Используйте виртуальное окружение Python:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r ml/requirements.txt
```

**Порты заняты?**
- Измените порты в `server/server.js` и `vite.config.ts`