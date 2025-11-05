import pandas as pd
import sys

#   Definición variables

discretes = [   #   Todas las columnas discretas del dataset
    'gender', 'ethnicity', 'education_level', 'income_level',
    'employment_status', 'smoking_status', 'diabetes_stage'
]
continuous = [  #   Todas las columnas continuas del dataset
    'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week',
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day',
    'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides',
    'glucose_fasting', 'glucose_postprandial', 'insulin_level', 'hba1c',
    'diabetes_risk_score'
]
binaries = [    #   Todas las columnas binarias del dataset
    'family_history_diabetes', 'hypertension_history',
    'cardiovascular_history', 'diagnosed_diabetes'
]
#   Ruta a dataset original, en caso de estar en otra ruta se debe cambiar aquí
dataset_path = './csv/diabetes_dataset.csv'

#   Funciones

def analyze_dataset(ds):    #   Análisis de dataset, datos iniciales de este
    print("=" * 80)
    print(f"Iniciando analisis...\nTotal de filas: {len(ds)}\nTotal de columnas: {len(ds.columns)}")
    print("-" * 60)
    print("Verificación de datos nulos")
    null_counts = ds.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls == 0:
        print("No se encontraron valores nulos en ninguna columna :D")
    else:
        print(f"Se encontraron {total_nulls} valores nulos\nColumnas con nulos:")
        print(null_counts[null_counts > 0])
    print("-" * 60)
    print("Analisis de columnas discretas")
    for col in discretes:
        try:
            discrete_values = ds[col].unique()
            print(f"Valores de '{col}': {list(discrete_values)}")
        except Exception as e:
            print(f"Eror analizando columna '{col}': {e}")
    print("-" * 60)
    print("Analisis de columnas continuas")
    for col in continuous:
        try:
            numeric_col = pd.to_numeric(ds[col], errors='coerce')
            if numeric_col.isnull().all():
                print(f"Columna '{col}':")
                print("    ERROR: No se pudieron encontrar datos numericos (podria contener solo texto o nulos)")
            else:
                col_min = numeric_col.min()
                col_max = numeric_col.max()
                print(f"Columna '{col}': [{col_min}, {col_max}]")
        except Exception as e:
            print(f"Error analizando la columna '{col}': {e}")
    print("-" * 60)
    print("Analisis de columnas binarias")
    for col in binaries:
        try:
            binaries_col = pd.to_numeric(ds[col], errors='coerce')
            col_min = binaries_col.min()
            col_max = binaries_col.max()
            if binaries_col.isnull().all():
                print(f"Columna '{col}':")
                print("    ERROR: No se pudieron encontrar datos numericos (podria contener solo texto o nulos)")
            elif col_min != 0 or col_max != 1:
                print(f"Columna '{col}': No binaria, rango [{col_min}, {col_max}]")
            else:
                print(f"Columna '{col}' correcta")
        except Exception as e:
            print(f"Error analizando la columna '{col}': {e}")
    print("-" * 60)
    print("Porcentaje de aparición para 'diabetes_stage'")
    stage_percentages = ds['diabetes_stage'].value_counts(normalize=True) * 100
    if stage_percentages.empty:
        print("Columna vacia...")
    else:
        for stage, percentage in stage_percentages.items():
            print(f"  {stage}: {percentage}%")
        nulls_stage = ds['diabetes_stage'].isnull().sum()
        if nulls_stage > 0:
            print(f"  Valores nulos: {nulls_stage}, filas ignoradas para porcentaje")
    print("Analisis finalizado...")
    print("=" * 80)

def filter_dataset(ds):
    print("=" * 100)
    print("Inicializando filtrado de dataset...")
    print("-" * 60)
    print("Eliminado de 15 mil filas 'diabetes_stage' = 'type 2'")
    df_type_2 = ds[ds['diabetes_stage'] == 'Type 2']
    df_to_drop = df_type_2.sample(n=15000, random_state=42)
    indices_to_drop = df_to_drop.index
    ds = ds.drop(indices_to_drop)
    print("Se eliminaron 15000 filas con 'diabetes_stage' = 'Type 2'")
    print("-" * 60)
    print("Eliminando columnas")
    cols_to_drop = [
        'ethnicity', 'education_level', 'income_level', 
        'employment_status', 'diabetes_risk_score', 'diagnosed_diabetes'
    ]
    ds = ds.drop(columns=cols_to_drop)
    for col in cols_to_drop:
        if col in discretes:
            discretes.remove(col)
        elif col in continuous:
            continuous.remove(col)
        elif col in binaries:
            binaries.remove(col)
    print(f"Columnas eliminadas por no necesidad de uso: {list(cols_to_drop)}")
    print("-" * 60)
    print("Cambiando valores discretos a continuos")
    #   Cambio de variables para gender
    gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
    ds['gender'] = ds['gender'].map(gender_map)
    discretes.remove('gender')
    continuous.append('gender')
    print("Columna 'gender' mapeada: {'Male': 0, 'Female': 1, 'Other': 2}")
    #   Cambio de variables para somking_status
    smoking_map = {'Never': 0, 'Former': 1, 'Current': 2}
    ds['smoking_status'] = ds['smoking_status'].map(smoking_map)
    discretes.remove('smoking_status')
    continuous.append('smoking_status')
    print("Columna 'smoking_status' mapeada: {'Never': 0, 'Former': 1, 'Current': 2}")
    #   Cambio de variables para diabetes_stage
    diabetes_map = {
        'No Diabetes': 0, 
        'Pre-Diabetes': 1, 
        'Type 1': 2,
        'Type 2': 3, 
        'Gestational': 4
    }
    ds['diabetes_stage'] = ds['diabetes_stage'].map(diabetes_map)
    discretes.remove('diabetes_stage')
    continuous.append('diabetes_stage')
    print("Columna 'diabetes_stage' mapeada: {'No Diabetes': 0, 'Pre-Diabetes': 1, 'Type 1': 2, 'Type 2': 3, 'Gestational': 4}")
    print("-" * 60)
    print("Datos post-filtrado y limpiado")
    analyze_dataset(ds)
    modified_dataset_path = './csv/filtered_dataset.csv'    #   Ruta de salida
    print(f"Guardando dataset modificado en {modified_dataset_path}")
    try:
        ds.to_csv(modified_dataset_path, index=False)
        print("Dataset guardado correctamente!")
    except Exception as e:
        print(f"Error al guardar archivo en {modified_dataset_path}: {e}")
    print("=" * 100)

#   Main
print(f"Iniciando análisis y filtrado de {dataset_path}")
try:
    ds = pd.read_csv(dataset_path)
    analyze_dataset(ds)
    filter_dataset(ds)
except pd.errors.EmptyDataError as e:
    print("\n===== ERROR CON CSV =====\n")
    raise ValueError(f"Dataset '{dataset_path}' esta vacio: {e}")
except Exception as e:
    print(f"\n===== ERROR INESPERADO =====\nError al leer o procesar dataset: {e}")
    sys.exit(1)    