import pandas as pd
from sklearn.model_selection import train_test_split

dataset_path = './csv/filtered_dataset.csv'

#   Leer y separar dataset
ds = pd.read_csv(dataset_path)
x = ds.drop('diabetes_stage')
y = ds['diabetes_stage']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#   Leer archivo de configuración
