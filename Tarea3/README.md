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

- _tensorflow_ es la librería que permite crear redes neuronales convolucionales (_CNN_):
```
pip install tensorflow
```
* _tqdm_ para barras de progreso durante la ejecución:
```
pip install tqdm
```
+ Finalmente se instalan las librerías _Matplotlib_, _Seaborn_ y _Scikit-Learn_ para los análisis finales y gráficas:
```
pip install matplotlib seaborn scikit-learn
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

El mejor porcentaje de dropout encontrado fue 0.2 (o 20%), por lo tanto es el que se utilizará para la arquitectura final, resultando en:

- _batch size_: 128.
* _epochs_: 20.
+ _learning rate_: 0.0001.
- _kernel size_: (5, 5).
* _architecture_ (cantidad de filtros en capa de convolución): 32, 64 y 128.
+ _dropout_: 0.2

## Comparación

Finalmente se tienen dos códigos, el primero es [final_train.py](./final_train.py), el cual se encarga de crear y entrenar los dos modelos (con y sin dropout) con la arquitectura e hiperparámetros definidos anteriormente, una vez entrenados se guardan los modelos y su historial de entrenamiento (para tener _accuracy_ y _loss_ por época) en [final_models](./final_models/).

En cuanto a la comparación a realizar se encuentra toda hecha en [compare.py](./compare.py), en donde utilizando el conjunto de testing separado inicialmente se calculan las predicciones de cada modelo, para luego calcular el _accuracy_, _f1 score_ y _confusion matrix_ de cada modelo, imprimiendo estos valores en una tabla comparativa, luego se grafican las curvas de _Loss_ y _Accuracy_ del entrenamiento de los modelos (con los conjuntos de datos de entrenamiento y validación), finalmente se crea un gráfico de calor de la _confusion matrix_.

De esta manera entonces se está comparando las evolución del entrenamiento con el _accuracy_ y _loss_ por época del conjunto de entrenamiento y validación, entregandonos una gran comparación de como avanza el entrenamiento de cada modelo. Por otro lado, utilizando el conjunto de testing se obtienen las métricas. _Accuracy_, la cual es la cantidad de aciertos que tiene el modelo respecto de la etiqueta real. _A1 score_ es una medida más robusta que solo el porcentaje de aciertos obtenidos, esta mide qué tantas veces el modelo no etiquetó correctamente algo y a la vez qué tantas veces etiquetó algo mal, de esta manera si se tiene un _f1 score_ cercano al accuracy es un modelo bueno, pero si está muy bajo significa que tiene muchos falsos positivos o negativos. Y finalmente se escogió la _confusion matrix_ extra para terminar de medir bien los modelos y poder dar un correcto análisis de estos, esta métrica entega información más detallada de lo que ocurre con las predicciones, esta es una matriz con cada posible valor y los valores de la predicción, siendo la diagonal las veces que el modelo acertó correctamente y el resto de la fila las veces que el modelo predijo erróneamente una etiqueta con la otra (específicamente).

## Declaración de uso de IA

En cuanto al uso de IA generativa para esta tarea, fue principalmente con el objetivo de investigar, solucionar errores de sintaxis y uso de librerías. En cuanto al uso de librerías fue para entender el uso de _tensorflow_ y la creación de los modelos, su uso, etc. También con otras más externas, por ejemplo el uso de _tqdm_ para crear las barras de progreso en el código y poder monitorear el avance de los códigos o el uso de _matplotlib_ y sus cercanas para la creación de los gráficos. En cuanto a la investigación fue principalmente para entender qué opciones se tenían a la hora de programar todo, librerías, etc. o por ejemplo en la comparación investigar qué métricas se podían usar, una vez se tenía la lista de las candidatas se inició una investigación por cuenta propia para terminar el trabajo (nunca se usó para análisis o utilización en sí de la librería).