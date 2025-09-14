#!/bin/bash

# Скрипт для запуска системы с улучшенной моделью
echo "🚗 Запуск inDrive Car Analysis с улучшенной моделью"
echo "=================================================="

# Проверяем наличие Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден. Установите Python3 для продолжения."
    exit 1
fi

# Проверяем наличие Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js не найден. Установите Node.js для продолжения."
    exit 1
fi

# Проверяем наличие npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm не найден. Установите npm для продолжения."
    exit 1
fi

echo "✅ Все зависимости найдены"

# Устанавливаем зависимости для Python (если нужно)
echo "📦 Проверяем Python зависимости..."
cd model
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
fi
cd ..

# Устанавливаем зависимости для Node.js (если нужно)
echo "📦 Проверяем Node.js зависимости..."
cd server
if [ -f "package.json" ]; then
    npm install
fi
cd ..

# Устанавливаем зависимости для фронтенда
echo "📦 Проверяем фронтенд зависимости..."
if [ -f "package.json" ]; then
    npm install
fi

echo ""
echo "🚀 Запускаем систему..."

# Запускаем Python API в фоне
echo "🐍 Запуск Python API (порт 5010)..."
cd model
python3 improved_app.py &
PYTHON_PID=$!
cd ..

# Ждем немного, чтобы API запустился
sleep 3

# Запускаем фронтенд
echo "⚛️  Запуск React фронтенда (порт 5173)..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Система запущена!"
echo "🐍 Python API: http://localhost:5010"
echo "⚛️  React фронтенд: http://localhost:5173"
echo "🤖 Используется улучшенная модель из папки model/"
echo ""
echo "Для остановки нажмите Ctrl+C"

# Функция для корректного завершения
cleanup() {
    echo ""
    echo "🛑 Остановка системы..."
    kill $PYTHON_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Система остановлена"
    exit 0
}

# Обработка сигнала завершения
trap cleanup SIGINT SIGTERM

# Ждем завершения
wait
