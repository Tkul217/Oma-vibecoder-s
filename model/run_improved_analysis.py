#!/usr/bin/env python3
"""
Скрипт для запуска улучшенного анализа автомобилей
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command, description):
    """Запуск команды с описанием"""
    print(f"\n🔄 {description}")
    print(f"Команда: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Успешно выполнено")
        if result.stdout:
            print("Вывод:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка: {e}")
        if e.stdout:
            print("Вывод:", e.stdout)
        if e.stderr:
            print("Ошибки:", e.stderr)
        return False

def check_requirements():
    """Проверка требований"""
    print("🔍 Проверка требований...")
    
    required_files = [
        'improved_car_model.py',
        'improved_train.py', 
        'improved_app.py',
        'prepare_4folders.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Отсутствуют файлы: {missing_files}")
        return False
    
    print("✅ Все файлы на месте")
    return True

def main():
    print("🚗 inDrive Advanced Car Analysis - Автоматический запуск")
    print("=" * 60)
    
    # Проверяем требования
    if not check_requirements():
        print("❌ Не все требования выполнены")
        return False
    
    # Шаг 1: Подготовка данных
    print("\n📊 ШАГ 1: Подготовка данных")
    if not run_command("python prepare_4folders.py", "Подготовка данных из 4 папок"):
        print("❌ Ошибка подготовки данных")
        return False
    
    # Проверяем что данные созданы
    if not os.path.exists('data/dataset.csv'):
        print("❌ Файл dataset.csv не создан")
        return False
    
    # Шаг 2: Обучение модели
    print("\n🤖 ШАГ 2: Обучение улучшенной модели")
    if not run_command("python improved_train.py", "Обучение улучшенной модели"):
        print("❌ Ошибка обучения модели")
        return False
    
    # Проверяем что модель создана
    if not os.path.exists('best_improved_model.pth'):
        print("❌ Модель не создана")
        return False
    
    # Шаг 3: Запуск API
    print("\n🌐 ШАГ 3: Запуск улучшенного API")
    print("Запускаем улучшенный API сервер...")
    print("=" * 60)
    print("🚀 API будет доступен по адресу: http://localhost:5000")
    print("📱 Веб-интерфейс: http://localhost:5000")
    print("📡 API документация: http://localhost:5000/info")
    print("❤️  Проверка здоровья: http://localhost:5000/health")
    print("=" * 60)
    print("Нажмите Ctrl+C для остановки сервера")
    print("=" * 60)
    
    try:
        # Запускаем API
        subprocess.run("python improved_app.py", shell=True, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Сервер остановлен пользователем")
    except Exception as e:
        print(f"❌ Ошибка запуска API: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 Все готово! Улучшенный анализ автомобилей запущен")
    else:
        print("\n❌ Произошла ошибка при запуске")
        sys.exit(1)
