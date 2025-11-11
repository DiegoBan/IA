import pandas as pd
import concurrent.futures
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

#   Función para leer txt de configuración para modelos
def leer_config(ruta):
    RL_config = []
    SVM_config = []
    try:
        with open(ruta, 'r') as f:
            def leer_bloque(file):
                cantidad = file.readline()
                if not cantidad:
                    return None
                try:
                    num_lineas = int(cantidad.strip())
                except ValueError:
                    print(f"Se esperaba numero, se encontro: {cantidad.strip()}")
                    return []
                if num_lineas < 3:
                    print("Se esperan al menos 3 instancias por tecnica")
                    return []
                bloque_actual = []
                for _ in range(num_lineas):
                    fila_datos = file.readline()
                    if not fila_datos:
                        print(f"Advertencia, fin de archivo inesperado, se esperaban {num_lineas} filas.")
                        break
                    try:
                        partes = fila_datos.strip().split()
                        val1 = int(partes[0])
                        val2 = float(partes[1])
                        bloque_actual.append((val1, val2))
                    except (ValueError, IndexError):
                        print(f"Advertencia: Omitiendo linea mal formada: {fila_datos.strip()}")
                return bloque_actual
            RL_config = leer_bloque(f)
            if RL_config is None:
                print("Error: Archivo de configuracion vacio")
                return [], []
            SVM_config = leer_bloque(f)
            if SVM_config is None:
                print("Error: Archivo de configuracion vacio")
                return RL_config, []
    except FileNotFoundError:
        print(f"No se encontro el archivo en '{ruta}'")
        return [], []
    except Exception as e:
        print(f"Ocurrio error inesperado: {e}")
        return [], []
    return RL_config, SVM_config

def get_batch(data_x, data_y, batch_size, start):
    data_size = len(data_x)
    if data_size != len(data_y):
        raise ValueError("data_x y data_y no tienen el mismo largo")
    while True:
        end = start + batch_size
        if end <= data_size:
            yield (data_x[start:end], data_y[start:end], end)
        else:
            parte1_x = data_x[start:data_size]
            parte1_y = data_y[start:data_size]
            faltantes = end - data_size
            parte2_x = data_x[0:faltantes]
            parte2_y = data_y[0:faltantes]
            if isinstance(data_x, (pd.DataFrame, pd.Series)):
                batch_x = pd.concat([parte1_x, parte2_x], axis=0, ignore_index=True)
            else:
                batch_x = parte1_x + parte2_x
            if isinstance(data_y, (pd.DataFrame, pd.Series)):
                batch_y = pd.concat([parte1_y, parte2_y], axis=0, ignore_index=True)
            else:
                batch_y = parte1_y + parte2_y
            yield (batch_x, batch_y, end%data_size)
        start = end % data_size

def entrenamiento(modelo, x_train, y_train, clases):
    batch_gen = get_batch(x_train, y_train, modelo['batch_size'], modelo['batch_index'])
    for _ in range(5):
        x_batch, y_batch, modelo['batch_index'] = next(batch_gen)
        modelo['model'].partial_fit(x_batch, y_batch, classes=clases)
    modelo['score'] = f1_score(y_train, modelo['model'].predict(x_train), average='macro')
    return modelo

if __name__ == "__main__":
    dataset_path = './csv/filtered_dataset.csv'
    config_path = './config_p2.txt'
    #   Leer y separar dataset
    print("===== Leyendo y separando dataset =====")
    ds = pd.read_csv(dataset_path)
    x = ds.drop('diabetes_stage', axis=1)
    y = ds['diabetes_stage']
    clases = y.unique()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    #   Leer archivo de configuración
    RL_config, SVM_config = leer_config(config_path)
    print("===== Configuracion leida =====")
    print(f"Configuracion de RL: {RL_config}")
    print(f"Configuracion de SVM: {SVM_config}")
    #   Modelos a utilizar
    modelos = {}
    id_contador = 0
    for model_config in RL_config:
        nombre = f"RL_{id_contador}"
        modelos[nombre] = {
            'model': SGDClassifier(loss='log_loss', learning_rate='constant', eta0=model_config[1]),
            'batch_size': model_config[0],
            'score': 0.0,
            'batch_index': 0
        }
        id_contador += 1
    id_contador = 0
    for model_config in SVM_config:
        nombre = f"SVM_{id_contador}"
        modelos[nombre] = {
            'model': SGDClassifier(loss='hinge', learning_rate='constant', eta0=model_config[1]),
            'batch_size': model_config[0],
            'score': 0.0,
            'batch_index': 0
        }
    #   Entrenamiento
    print("==== Iniciando entrenamientos... =====")
    while len(modelos) > 2:
        print("-- Entreno --")
        futuros = {}
        peor_modelo = None
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for nombre, m in modelos.items():
                print(f"Entrenando {nombre}")
                futuro = executor.submit(
                    entrenamiento,
                    m,
                    x_train,
                    y_train,
                    clases
                )
                futuros[nombre] = futuro
            print("-- Esperando resultados --")
            for nombre, futuro in futuros.items():
                try:
                    modelos[nombre] = futuro.result()
                    if peor_modelo is None:
                        peor_modelo = nombre
                    elif modelos[nombre]['score'] < modelos[peor_modelo]['score']:
                        peor_modelo = nombre
                except Exception as e:
                    print(f"Error entrenando {nombre}: {e}")
        print(f"Se elimina el peor modelo {peor_modelo}, score = {modelos[peor_modelo]['score']:.4f}")
        modelos.pop(peor_modelo)
    #   Resultados finales
    print("===== Mejores Modelos: =====")
    for nombre, m in modelos.items():
        print(f"Modelo {nombre}, con un porcentaje de aciertos: {accuracy_score(y_test, m['model'].predict(x_test))*100}%")