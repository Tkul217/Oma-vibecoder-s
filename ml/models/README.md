# Модели машинного обучения

Поместите ваши обученные PyTorch модели в эту папку:

## Требуемые файлы:

- `cleanliness_model.pth` - модель для определения чистоты автомобиля
- `condition_model.pth` - модель для определения повреждений

## Формат моделей:

Модели должны быть сохранены с помощью `torch.save()`:

```python
# Пример сохранения модели
torch.save(model.state_dict(), 'cleanliness_model.pth')
torch.save(model.state_dict(), 'condition_model.pth')
```

## Архитектура моделей:

- Входной размер: 224x224x3 (RGB изображение)
- Выходной размер: 2 класса (binary classification)
- Рекомендуемая архитектура: ResNet50 или EfficientNet

## Классы:

### Cleanliness Model:
- Класс 0: clean (чистый)
- Класс 1: dirty (грязный)

### Condition Model:
- Класс 0: intact (целый)
- Класс 1: damaged (поврежденный)