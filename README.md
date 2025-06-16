# Инжиниринг данных. Автоматизация и оркестрация пайплайна машинного обучения с использованием Apache Airflow и облачного хранилища.

## Описание проекта

Цель проекта — построить полный ETL-процесс и обучить модель машинного обучения на датасете breast_cancer.csv, автоматизировав все этапы с помощью Apache Airflow. Результат - устойчивая система, в которой реализован контроль версий, повторяемость и возможность масштабирования.

---

## Структура проекта

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

---

## Используемые технологии

- Python 3.10+
- Apache Airflow
- pandas, scikit-learn, joblib
- bash + argparse
- JSON / CSV / PKL

---

## Этапы пайплайна

1. Загрузка (load_data.py) - чтение CSV, сохранение копии и генерация отчёта.
2. Предобработка (preprocess.py) - очистка, кодирование меток, стандартизация.
3. Обучение (train_model.py) - деление на выборки, обучение логистической регрессии, сохранение в pkl.
4. Метрики (metrics.py) - предсказания и расчёт метрик (accuracy, F1).
5. Финализация (save_results.py) - упаковка всех результатов в results/final/.

---

## DAG-файл: (pipeline_dag.py)

Настроен в Airflow на выполнение по требованию (schedule: `None`). Включает 5 связанных задач одна за другой.

```bash
airflow dags trigger ml_pipeline_dag
```

---

## Метрики модели

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

## Возможные улучшения

- Добавить телеграм-уведомления в DAG
- Сохранение артефактов в облако (Яндекс.Диск / Google Drive)
- Разделение train/test через Cross-Validation
- Использование MLflow для логирования метрик
- Миграция с BashOperator → PythonOperator
