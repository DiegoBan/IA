# Tarea 2

## Preparación

### Entorno

Para la ejecución de esta tarea primero se creó un entorno virtual para evitar que el uso de liberías de python vaya a corromper tal instalación local, entonces, se crea el entorno:
```
cd ruta/de/carpeta
python3 -m venv venv
```
Para entrar (activar) entorno se ejecuta:
```
source venv/bin/activate
```
En caso de querer salir del entorno virtual, se debe ejecutar:
```
deactivate
```

### Librerías

En esta sección se detallan las librerías que utilizarán los códigos, las cuales es necesario tener instaladas en el sistema o en el entorno virtual de python en caso de estar utilizandolo.

- pandas: para analisis y filtrado de dataset.
```
pip install pandas
```
* scikit-learn: Para uso de modelos de ML.
```
pip install scikit-learn
```

## Dataset

El dataset escogido es '[Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset)' (en este link se puede encontrar el desglose de cada columna y lo que representa) el cual se encuentra en [diabetes_dataset.csv](csv/diabetes_dataset.csv) y cuenta con 31 columnas, dentro de las cuales 3 son para predición.
Se limpiarán y eliminarán columnas no deseadas, esto a través del código [dataset_fix.py](dataset_fix.py), el cual inicialmente muestra información del dataset original sin ningún cambio, para luego filtrarla.
El filtrado consta de 3 principales secciones, primero elimina 15 mil filas aleatorias con 'diabetes_stage' igual a 'Type 2', esto más que nada para tener una relación entre las categorías del dataset no tan desigual, luego elimina las columnas 'ethnicity', 'education_level', 'income_level' y 'employment_status' ya que no presentan una gran relación con el 'diabetes_stage' que es lo que se busca predecir, por lo tanto podrían solo introducir ruido a nuestros modelos, también se eliminaron las columnas 'diabetes_risk_score' y 'diagnosed_diabetes' ya que son para otro tipo de predicciones que no estámos buscando, y la última sección convierte las columnas discretas 'gender', 'somking_status', y 'diabetes_stage' en columnas continuas, mapeando cada uno de sus valores a uno numérico.
y finalmente guarda el dataset que se utilizará para la realización de los códigos solicitados de la tarea en [filtered_dataset.csv](csv/filtered_dataset.csv).

## Parte 1



## Parte 2

En esta segunda parte se implementa a través de SGDClassifier modelos de Regresión Logística ('log_loss') y SVM ('higne') que se entrenan con el mismo dataset anteriormente utilizado, se crean distintas instancias de estos modelos con distintos hiperparámetros para variar su funcionamiento, de esta manera se pueden estudiar los cambios generados por modificarlos. Estos modelos se están entrenando y cada 5 épocas se comparan entre sí a través de la función f1_score, aquellas con menor score se eliminan hasta quedar con las mejores 2 configuraciones.
Finalmente los dos mejores modelos se evalúan con el conjunto del dataset separado para testing y así analizar sus resultados.

### archivo de configuración

El archivo de configuración es el que se encuentra en la misma ruta actual '[config_p2.txt](config_p2.txt)' y contiene la siguiente configuración de ejemplo:
```
3 -> cantidad de configuraciones para Regresión Lineal
10000 0.1 -> batch size y taza de aprendizaje (alpha) de primera instancia de RL
5000 0.5 -> batch size y taza de aprendizaje de segunda instancia
15000 0.01 -> batch size y ... de tercera instancia
3 -> cantidad de configuraciones para SVM
10000 0.1 -> batch size y taza de aprendizaje (alpha) de primera instancia de SVM
5000 0.5 -> batch size y taza de aprendizaje de segunda instancia
15000 0.01 -> batch size y ... de tercera instancia
```