from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='ml_pipeline_dag',
    default_args=default_args,
    description='ETL + ML pipeline with Airflow',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:

    load_data = BashOperator(
        task_id='load_data',
        bash_command='python "/Users/ekaterina.dorokhova/Desktop/Инжиниринг данных/Экзамен/DE_final_project/airflow_home/etl/load_data.py" '
                     '--input "/Users/ekaterina.dorokhova/Desktop/Инжиниринг данных/Экзамен/DE_final_project/airflow_home/data/breast_cancer.csv" '
                     '--output "/Users/ekaterina.dorokhova/Desktop/Инжиниринг данных/Экзамен/DE_final_project/airflow_home/results"'
    )

    preprocess = BashOperator(
        task_id='preprocess',
        bash_command='python "/Users/ekaterina.dorokhova/Desktop/Инжиниринг данных/Экзамен/DE_final_project/airflow_home/etl/preprocess.py" '
                     '--input "/Users/ekaterina.dorokhova/Desktop/Инжиниринг данных/Экзамен/DE_final_project/airflow_home/results/raw_data.csv" '
                     '--output "/Users/ekaterina.dorokhova/Desktop/Инжиниринг данных/Экзамен/DE_final_project/airflow_home/results"'
    )

    train_model = BashOperator(
        task_id='train_model',
        bash_command='python "/Users/ekaterina.dorokhova/Desktop/Инжиниринг данных/Экзамен/DE_final_project/airflow_home/etl/train_model.py" '
                     '--x "/Users/ekaterina.dorokhova/Desktop/Инжиниринг данных/Экзамен/DE_final_project/airflow_home/results/X_processed.csv" '
                     '--y "/Users/ekaterina.dorokhova/Desktop/Инжиниринг данных/Экзамен/DE_final_project/airflow_home/results/y.csv" '
                     '--output "/Users/ekaterina.dorokhova/Desktop/Инжиниринг данных/Экзамен/DE_final_project/airflow_home/results"'
    )

    calc_metrics = BashOperator(
        task_id='calc_metrics',
        bash_command='python "/Users/ekaterina.dorokhova/Desktop/Инжиниринг данных/Экзамен/DE_final_project/airflow_home/etl/metrics.py" '
                     '--x "/Users/ekaterina.dorokhova/Desktop/Инжиниринг данных/Экзамен/DE_final_project/airflow_home/results/X_processed.csv" '
                     '--y "/Users/ekaterina.dorokhova/Desktop/Инжиниринг данных/Экзамен/DE_final_project/airflow_home/results/y.csv" '
                     '--model "/Users/ekaterina.dorokhova/Desktop/Инжиниринг данных/Экзамен/DE_final_project/airflow_home/results/model.pkl" '
                     '--output "/Users/ekaterina.dorokhova/Desktop/Инжиниринг данных/Экзамен/DE_final_project/airflow_home/results"'
    )

    save_results = BashOperator(
        task_id='save_results',
        bash_command='python "/Users/ekaterina.dorokhova/Desktop/Инжиниринг данных/Экзамен/DE_final_project/airflow_home/etl/save_results.py" '
                     '--source "/Users/ekaterina.dorokhova/Desktop/Инжиниринг данных/Экзамен/DE_final_project/airflow_home/results" '
                     '--output "/Users/ekaterina.dorokhova/Desktop/Инжиниринг данных/Экзамен/DE_final_project/airflow_home/results/final"'
    )

    # Зависимости
    load_data >> preprocess >> train_model >> calc_metrics >> save_results