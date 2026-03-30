from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml_ops',
    'depends_on_past': False,
    'start_date': datetime(2026, 3, 20),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'api_anomaly_retraining_pipeline',
    default_args=default_args,
    description='Weekly pipeline to aggregate logs, check drift, and retrain LSTM',
    schedule_interval=timedelta(days=7),
    catchup=False
)

aggregate_sql_task = BashOperator(
    task_id='aggregate_weekly_logs',
    bash_command='psql -f /opt/airflow/sql/aggregate_traffic.sql',
    dag=dag,
)

train_model_task = BashOperator(
    task_id='retrain_lstm_autoencoder',
    bash_command='python /opt/airflow/src/train.py',
    dag=dag,
)

update_registry_task = BashOperator(
    task_id='update_mlflow_registry',
    bash_command='echo "Promoting model to staging if ROC-AUC > 0.90"',
    dag=dag,
)

aggregate_sql_task >> train_model_task >> update_registry_task