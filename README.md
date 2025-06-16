# Инжиниринг данных. Автоматизация и оркестрация пайплайна машинного обучения с использованием Apache Airflow и облачного хранилища.

**Описание проекта**

Цель проекта — построить полный ETL-процесс и обучить модель машинного обучения на датасете breast_cancer.csv, автоматизировав все этапы с помощью Apache Airflow. Результат - устойчивая система, в которой реализован контроль версий, повторяемость и возможность масштабирования.

**Структура проекта**

```
project/
│
├── data/
│   └── breast_cancer.csv # Исходные данные
│
├── etl/
│   ├── load_data.py # Загрузка и EDA
│   ├── preprocess.py # Очистка и масштабирование
│   ├── train_model.py # Обучение модели
│   ├── metrics.py # Расчёт метрик
│   └── save_results.py # Финальная упаковка
│
├── results/
│   ├── X_processed.csv
│   ├── y.csv
│   ├── model.pkl
│   ├── metrics.json
│   ├── eda_report.txt
│   └── final/ # Финальные артефакты
│
├── dags/
│   └── pipeline_dag.py # DAG-файл Airflow
│
└── README.md # Описание проекта
```

**Используемые технологии**

- Python 3.10+
- Apache Airflow
- pandas, scikit-learn, joblib
- bash + argparse
- JSON / CSV / PKL

**Этапы пайплайна**

1. Загрузка (load_data.py) - чтение CSV, сохранение копии и генерация отчёта.
2. Предобработка (preprocess.py) - очистка, кодирование меток, стандартизация.
3. Обучение (train_model.py) - деление на выборки, обучение логистической регрессии, сохранение в pkl.
4. Метрики (metrics.py) - предсказания и расчёт метрик (accuracy, F1).
5. Финализация (save_results.py) - упаковка всех результатов в results/final/.

**DAG-файл: (pipeline_dag.py)**

Настроен в Airflow на выполнение по требованию (schedule: `None`). Включает 5 связанных задач одна за другой.

```bash
airflow dags trigger ml_pipeline_dag
```

**Метрики модели**

Пример содержимого metrics.json:

```json
{
    "accuracy": 0.95,
    "precision": 0.96,
    "recall": 0.93,
    "f1_score": 0.94
}
```

---
## Этап 1. Разработка ETL-компонентов

### Скрипт 1 - загрузка и первичный анализ данных (load_data.py)

**Цель скрипта:**
- Загрузить CSV-файл с данными (Breast Cancer Wisconsin Diagnostic)
- Выполнить базовый EDA: размер, пропущенные значения, типы колонок
- Сохранить отчёт EDA и (опционально) копию исходного датасета

**Структура** (рекомендуемая по учебным материалам)**
- Аргументы через `argparse`
- Минимальный логгинг (`print()` или `logging`)
- Обработка ошибок
- Воспроизводимость

**Что делает скрипт:**
- Загружает CSV
- Пишет краткий EDA-отчет в `results/eda_report.txt`
- Сохраняет копию CSV в `results/raw_data.csv`

**Код etl/load_data.py**

```python
import pandas as pd
import os
import argparse
import logging

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Датасет загружен: {file_path}")
        return df
    except Exception as e:
        logging.error(f"Ошибка при загрузке CSV: {e}")
        raise

def basic_eda(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    shape_info = f"Размерность: {df.shape}"
    nulls_info = df.isnull().sum().to_string()
    dtypes_info = df.dtypes.to_string()

    report = "\n".join([shape_info, "\nПропуски:\n", nulls_info, "\nТипы данных:\n", dtypes_info])

    report_path = os.path.join(output_dir, "eda_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    logging.info(f"EDA-отчёт сохранён в: {report_path}")
    return report_path

def save_copy(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "raw_data.csv")
    df.to_csv(output_path, index=False)
    logging.info(f"Копия датасета сохранена в: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load CSV and perform basic EDA")
    parser.add_argument("--input", required=True, help="Путь к CSV-файлу")
    parser.add_argument("--output", default="results", help="Папка для вывода отчёта")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df = load_csv(args.input)
    basic_eda(df, args.output)
    save_copy(df, args.output)
```

**Запуск вручную**

```bash
python etl/load_data.py --input data/breast_cancer.csv --output results/
```

---
## Этап 2. Предобработка

### Скрипт 2

---
## Этап 3. Обучение

### Скрипт 3

---
## Этап 4. Метрики

### Скрипт 4

---
## Этап 5. Сохранение результатов

### Скрипт 5

---
## Возможные улучшения

- Добавить телеграм-уведомления в DAG
- Сохранение артефактов в облако (Яндекс.Диск / Google Drive)
- Разделение train/test через Cross-Validation
- Использование MLflow для логирования метрик
- Миграция с BashOperator → PythonOperator
