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

**Работа скрипта:**
- Загрузка CSV-файла с данными (Breast Cancer Wisconsin Diagnostic),
- Выполнение базового EDA и написание отчёт в results/eda_report.txt (размер, пропущенные значения, типы колонок),
- Сохранение EDA-отчёта и копии исходного датасета в CSV в results/raw_data.csv.

**Структура**
- Аргументы через argparse
- Минимальный логгинг (print() или logging)
- Обработка ошибок
- Воспроизводимость

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

**Запуск (вручную)**

```bash
python etl/load_data.py --input data/breast_cancer.csv --output results/
```

---
## Этап 2. Предобработка данных

### Скрипт 2 - очистка и нормализация признаков (preprocess.py)

**Работа скрипта:**

1. Загрузка (raw_data.csv) - используются результат предыдущего шага
2. Очистка, удаление id, обработка пропусков
3. Преобразование категориальной переменной diagnosis в число (кодирование M в 1, B в 0)
4. Масштабирование числовых признаков (StandardScaler)
5. Сохранение предобработанных X (X_processed.csv) и y (y.csv) в папку results/.

**Код etl/preprocess.py**

```python
import pandas as pd
import numpy as np
import argparse
import os
import logging
from sklearn.preprocessing import StandardScaler

def preprocess(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_file)

    # Удаление неинформативного столбца
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Преобразование целевой переменной
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Делим на X и y
    y = df['diagnosis']
    X = df.drop(columns=['diagnosis'])

    # Масштабирование признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Сохранение результатов
    x_path = os.path.join(output_dir, "X_processed.csv")
    y_path = os.path.join(output_dir, "y.csv")
    X_scaled_df.to_csv(x_path, index=False)
    y.to_csv(y_path, index=False)

    logging.info(f"Файлы сохранены: {x_path}, {y_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Предобработка данных: очистка и масштабирование")
    parser.add_argument("--input", required=True, help="Путь к исходному CSV")
    parser.add_argument("--output", default="results", help="Путь для вывода")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    preprocess(args.input, args.output)
```

**Запуск (вручную)**

```bash
python etl/preprocess.py --input results/raw_data.csv --output results/
```

**Результат:**

- `results/X_processed.csv` — признаки после масштабирования,
- `results/y.csv` — целевая переменная числом.

---
## Этап 3. Обучение

### Скрипт 3

```python
```

**Работа скрипта:**

**Запуск**



---
## Этап 4. Метрики

### Скрипт 4

```python
```

**Работа скрипта:**

**Запуск**



---
## Этап 5. Сохранение результатов

### Скрипт 5

```python
```

**Работа скрипта:**

**Запуск**


---
## Возможные улучшения

- Добавить телеграм-уведомления в DAG
- Сохранение артефактов в облако (Яндекс.Диск / Google Drive)
- Разделение train/test через Cross-Validation
- Использование MLflow для логирования метрик
- Миграция с BashOperator → PythonOperator
