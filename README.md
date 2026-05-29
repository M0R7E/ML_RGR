# CS:GO bomb_planted — ML dashboard

Бинарная классификация: предсказание `bomb_planted` по состоянию раунда.

## Запуск

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python jobs\fit_all_models.py
streamlit run app\main.py
```

## Хранение артефактов

- модели: `storage/classifiers/*.pkl`
- реестр и метрики: `storage/classifiers/registry.txt` (текстовый формат)

## Структура

```
app/              # Streamlit-интерфейс (4 страницы)
backend/          # работа с данными и классификаторами
jobs/             # обучение моделей
storage/
  classifiers/    # сериализованные модели + registry.txt
data/
```
