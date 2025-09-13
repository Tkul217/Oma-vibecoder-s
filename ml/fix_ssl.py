#!/usr/bin/env python3
"""
Скрипт для исправления SSL проблем на macOS
Запустите перед обучением если есть проблемы с сертификатами
"""

import ssl
import certifi
import urllib.request

def fix_ssl_certificates():
    """Исправление SSL сертификатов"""
    
    print("🔧 Исправление SSL сертификатов...")
    
    # Создание контекста с правильными сертификатами
    context = ssl.create_default_context(cafile=certifi.where())
    
    # Установка как глобальный контекст
    ssl._create_default_https_context = lambda: context
    
    print("✅ SSL сертификаты исправлены!")
    
    # Тест загрузки
    try:
        print("🧪 Тестирование загрузки...")
        response = urllib.request.urlopen('https://download.pytorch.org/models/resnet50-0676ba61.pth', timeout=10)
        print(f"✅ Тест успешен! Размер файла: {len(response.read())} байт")
    except Exception as e:
        print(f"❌ Тест не прошел: {e}")
        print("💡 Попробуйте установить сертификаты:")
        print("   /Applications/Python\\ 3.x/Install\\ Certificates.command")

if __name__ == "__main__":
    fix_ssl_certificates()