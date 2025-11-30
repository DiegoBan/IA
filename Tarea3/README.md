# Tarea 3

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

- _tensorflow_ es la librería que permite crear redes neuronales convolucionales (_CNN_).
```
pip install tensorflow
```
* _tqdm_ para barras de progreso durante la ejecución
```
pip install tqdm
```

## Dataset

El dataset se debe separar en training, validation y testing, el total de las imagenes del dataset se encuentra en [img_total](./img_total/), este total se separa a través del código [img_divide](./img_divide.py) en 70% para training, 15 para validation y 15 para testing, los resultados son puestos en [img_divided](./img_divided/), dentro de esta carpeta está dividido en los porcentajes mencionados y dentro de estas están en las carpetas de cada grupo.

## Sintonización de parámetros

Para la búsqueda de los mejores hiperparámetros se tienen 2 códigos, [hiperparams_tuning1.py](./hiperparams_tuning1.py) sigue una estrategia greedy, probando distintas configuraciones de cada uno de los parámetros, guardando el indice del que obtiene mejores resultados (accuracy, porcentaje de aciertos con el set de validación), al final imprime los resultados.

El otro código es [hiperparams_tuning2.py](./hiperparams_tuning2.py), este realiza una busqueda completa de todas las combinaciones de parámetros entregados, entregando como resultado final aquella combinación y modelo con los mejores parámetros.

El modo de uso pensado es entregar la mayor cantidad de parámetros en el primer código (que es más eficiente realizando la búsqueda, pero menos exhaustivo), a partir de los resultados de este se agregan los parámetros de interés en el segundo código para que realice una busqueda de mayor profundidad, encontrando finalmente el modelo con los mejores parámetros.

### Parámetros sintonizados

Los parámetros sintonizados con el primer código fueron los siguientes:

- _batch size_: [16, 32, 64, 128]
* _epochs_: [5, 10, 15, 20]
+ _learning rate_: [0.001, 0.1, 0.01, 0.0001]
- _kernel size_: [(3, 3), (2, 2), (5, 5), (8, 8)]
* _architecture_ (cantidad de filtros en capa de convolución): 32 filtros, 32 y 64 filtros, 32, 64 y 128 filtros, 32, 64, 128 y 256 filtros.

A partir de los resultados de este código 1 se seleccionaron los siguientes parámetros para realizar la busqueda exhaustiva con el código 2:

- _batch size_: []
* _epochs_: []
+ _learning rate_: []
- _kernel size_: []
* _architecture_ (cantidad de filtros en capa de convolución): 

## Arquitectura base

A partir de los resultados obtenidos anteriormente se seleccionaron los siguientes parámetros para nuestra arquitectura base, ya que fueron con los que se obtuvieron mejores resultados:

- _batch size_:
* _epochs_:
+ _learning rate_:
- _kernel size_:
* _architecture_ (cantidad de filtros en capa de convolución): 

## Dropout

### Arquitectura con Dropout

## Comparación