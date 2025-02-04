# Filename: airflowMLOps.py
# Author: Nimalan Subramanian
# Created: 2025-02-02
# Description: Automates and orchestrates data preprocessing, model training, evaluation, and deployment, scheduled and managed by Apache Airflow.

##Data preprocessing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#load dataset
data = pd.read_csv('screentime_analysis.csv')

#check for missing values and duplicates
print(data.isnull().sum())
print(data.duplicated().sum())

#convert Date column to datetime and extract features
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month

#encode categorical 'App' column with one-hot encoding
data = pd.get_dummies(data, columns=['App'], drop_first=True)

#scale numerical features with MinMaxScaler
scaler = MinMaxScaler()
data[['Notifications', 'Times Opened']] = scaler.fit_transform(data[['Notifications', 'Times Opened']])

#feature engineering
data['Previous_Day_Usage'] = data['Usage (minutes)'].shift(1)
data['Notifications_x_TimesOpened'] = data['Notifications'] * data['Times Opened']

#save preprocessed data into file
data.to_csv('preprocessed_screentime_analysis.csv', index=False)

##Train Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#split data into features and target variable
X = data.drop(columns=['Usage (minutes)', 'Date'])
y = data['Usage (minutes)']

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

#evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

##Automate preprocessing with Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

#define data preprocessing function
def preprocess_data():
    file_path = 'screentime_analysis.csv'
    data = pd.read_csv(file_path)

    data['Date'] = pd.to_datetime(data['Date'])
    data['DayOfWeek'] = data['Date'].dt.day_of_week
    data['Month'] = data['Month'].dt.month

    data = data.drop(columns=['Date'])

    data = pd.get_dummies(data, columns=['App'], drop_first=True)

    scaler = MinMaxScaler()
    data[['Notifications', 'Times Opened']] = scaler.fit_transform(data[['Notifications', 'Times Opened']])

    preprocessed_path = 'preprocessed_screentime_analysis.csv'
    data.to_csv(preprocessed_path, index=False)
    print(f"Preprocessed data saved to {preprocessed_path}")

#define DAG
dag = DAG(
    dag_id='data_preprocessing',
    schedule_interval='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False,
)

#define task
preprocess_task = PythonOperator(
    task_id='preprocess',
    python_callable=preprocess_data,
    dag=dag,
)