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

## Ejecución

Una vez se tiene el entorno preparado con sus respectivas librerías se puede ejecutar cualquiera de los códigos en esta tarea de la misma forma:
```
python3 code.py
```

## Dataset

El dataset se debe separar en training, validation y testing, el total de las imagenes del dataset se encuentra en [img_total](./img_total/), este total se separa a través del código [img_divide](./img_divide.py) en 70% para training, 15 para validation y 15 para testing, los resultados son puestos en [img_divided](./img_divided/), dentro de esta carpeta está dividido en los porcentajes mencionados y dentro de estas están en las carpetas de cada grupo.

## Sintonización de parámetros

Para la búsqueda de los mejores hiperparámetros se tienen 2 códigos, [hiperparams_tuning1.py](./hiperparams_tuning1.py) sigue una estrategia greedy, probando distintas configuraciones de cada uno de los parámetros, guardando el indice del que obtiene mejores resultados (accuracy, porcentaje de aciertos con el set de validación), al final imprime los resultados.

El otro código es [hiperparams_tuning2.py](./hiperparams_tuning2.py), este realiza una busqueda completa de todas las combinaciones de parámetros entregados, entregando como resultado final aquella combinación y modelo con los mejores parámetros.

El modo de uso pensado es entregar la mayor cantidad de parámetros en el primer código (que es más eficiente realizando la búsqueda, pero menos exhaustivo), a partir de los resultados de este se agregan los parámetros de interés en el segundo código para que realice una busqueda de mayor profundidad, encontrando finalmente el modelo con los mejores parámetros.

### Parámetros sintonizados

Los parámetros sintonizados con el primer código fueron los siguientes:

- _batch size_: [16, 32, 64, 128].
* _epochs_: [5, 10, 15, 20].
+ _learning rate_: [0.001, 0.1, 0.01, 0.0001].
- _kernel size_: [(3, 3), (2, 2), (5, 5), (8, 8)].
* _architecture_ (cantidad de filtros en capa de convolución): 32 filtros, 32 y 64 filtros, 32, 64 y 128 filtros, 32, 64, 128 y 256 filtros.

A partir de los resultados de este código 1 se seleccionaron los siguientes parámetros para realizar la busqueda exhaustiva con el código 2, los seleccionados fueron principalmente aquellos que no tenían una diferencia considerable (mayor o cercano a 1%), por lo tanto aquellas opciones que sobresalían quedaron solas, mientras que aquellas que tenían una diferencia bastante menor fueron seleccionadas en su conjunto para ser probadas.

- _batch size_: [128].
* _epochs_: [15, 20].
+ _learning rate_: [0.0001].
- _kernel size_: [(5, 5)].
* _architecture_ (cantidad de filtros en capa de convolución): 32 y 64 filtros y 32, 64 y 128 filtros.

## Arquitectura base

A partir de los resultados obtenidos anteriormente se seleccionaron los siguientes parámetros para nuestra arquitectura base, ya que fueron con los que se obtuvieron mejores resultados:

- _batch size_: 128.
* _epochs_: 20.
+ _learning rate_: 0.0001
- _kernel size_: (5, 5)
* _architecture_ (cantidad de filtros en capa de convolución): 32, 64 y 128.

## Dropout

¿Qué es el dropout? Es una técnica que tiene como objetivo evitar el overfitting evitando que el modelo se memorice características del conjunto de entrenamiento, por ejemplo, algún pixel de algún color específico o bordes que se repiten solo en el conjunto de entrenamiento, esta capa se agrega al modelo y solo funciona a la hora de entrenar (fit), cuando se realizan predicciones se desactiva. Su funcionamiento es el siguiente, se le asigna un porcentaje (que según la literatura, este normalmente varía entre 0.2 y 0.5) que significa la probabilidad para cada neurona de apagarse durante una epoca de entrenamiento (i.e. su parámetro queda en 0 solo por esa época), este movimiento hace que se obligue al resto de neuronas a aprender bien las características más generales o robustas y no se centren en las demasiado específicas

### Arquitectura con Dropout

En cuanto a los parámetros a utilizar con dropout se utilizarán los mismos hiperparámetros del mejor modelo encontrado anteriormente, pero se debe agregar esta capa de dropout, a la cual también se le tiene que asignar un porcentaje (explicado anteriormente), para buscar qué porcentaje es mejor para el modelo y el aprendizaje se crea el código [dropout_tuning.py](./dropout_tuning.py), el cual busca el porcentaje con mayor acccuracy de entre los siguientes: [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5].

El mejor encontrado fue , por lo tanto es el que se utilizará para la arquitectura final, resultando en:

- _batch size_: 128.
* _epochs_: 20.
+ _learning rate_: 0.0001.
- _kernel size_: (5, 5).
* _architecture_ (cantidad de filtros en capa de convolución): 32, 64 y 128.
+ _dropout_:

## Comparación