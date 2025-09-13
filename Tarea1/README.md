# Tarea 1 - Inteligencia Artificial

En esta carpeta se desarrolla la tarea N°1 de Inteligencia Artificial, la cual consta de dos partes principales, a través de este README se explicará todo lo necesario para cada una de estas mismas y el resultado final almacenado en un Google Collab.

### Ejecución y utilización de python

Para ejecutar esto de una manera controlada y sin crear posibles errores por dependencias con otras librerías o funciones de python en nuestro computador (local), se utilizará un entorno virtual de python "_virtualenv_". Para la utilización de este entorno se tendrán los siguientes comandos de referencia:

1. Creación de entorno virtual
```
python3 -m venv .venv
```
2. Activar (o entrar) en tal entorno
```
source .venv/bin/activate
```
3. Para salir del entorno
```
desactivate
```
4. En caso de querer eliminar entorno (debe ejecutarse fuera de este)
```
rm -rf .venv
```
> [!CAUTION]
> Para el correcto funcionamiento de los siguientes códigos, se deben descargar las librerías correspondientes (contenidas en requirements.txt)
Para instalar las dependencias desde requirements.txt
```
pip install -r requirements.txt
```
> [!WARNING]
> Todos estos comandos hasta el momento solo han sido probados y ejecutados en una computadora con macOS y terminal zsh, puede que en otros sistemas operativos o terminales varíe un poco tal utilización de comandos.

## Parte 1

Para esta primera parte se utiliza la librería [pgmpy](https://pgmpy.org/index.html) y el dataset escogido fue ["Video Games Sales"](https://www.kaggle.com/datasets/gregorut/videogamesales) el cual es un ranking de los videojuegos más vendidos (última vez actualizado hace 9 años) y tiene 16598 datos. Con estos datos inicialmente aquellas columnas que son numéricas se discretizan (Año y ventas por región: "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales" y "Global_Sales") en grupos de igual rango, para posteriormente separar dataset en 2 grupos uno con el 70% de los datos y otro con el restante.

Una vez se tiene el dataset discretizado, se puede comenzar a trabajar con tales datos (esto ya que las redes bayesianas no funcionan con tantos datos distintos, las inferencias no serán correctas e intentarán predecir para cada dato númerico por separado, lo cual no entregará buenas probabilidades), la separación en 70% y 30% es para utilizar el 70% para entrenamiento y el resto para validar tales modelos predictivos.

Con _dataset70_ (que contiene el 70% de los datos) se crea _structure_ utilizando la función _HillClimbSearch_, este _structure_ contiene el objeto que es estructura báscia para crear un modelo, luego con esta estructura, la función _.estimate_ y entregandole un metodo de rankeo para el dataset se crean las dependencias del modelo, en este paso se crean 2 modelos distintos, uno utilizando el _scroing method_ *BIC (Bayesian Information Criterion)* y otro con *BDeu (distribution: dirichlet prior)*, y con estas dependencias a través de la función _DiscreteBayesianNetwork_ se crean los objetos en las variables _model1_ y _model2_ que si tienen la red conformada.

Con las redes conformadas con sus dependencias y todo lo necesario, se procede a calcular los parámetros y probabilidades de esta, a través de la función _.fit_, obteniendo parámetros para las dependencias identificadas por el metodo de aprendizaje de estructura.

Con los parámetros obtenidos para cada modelo, se pueden realizar distintas inferencias, a cada uno se le realizan 4 inferencias distintas entregandole evidencia, para mostrar su salida.

Finalmente se realiza una validación a cada modelo con el 30% de datos restantes los cuales habíamos separado para realizar justamente este paso, con ayuda de la IA _chatGPT_ se generaron en el csv _queries_inferences.csv_ que contiene todas las inferencias que se pueden realizar en ambos modelos a la vez, se realiza la inferencia por cada fila del 30%, comparando aquel dato con mayor probabilidad con el que realmente es proveniente de los datos, en caso de que si sea el mismo dato, se suma uno a _fav\_cases_, mientras que en todas las iteraciones se suma uno a _total\_cases_, creando un porcentaje de acierto para cada modelo.

## Parte 2

En esta siguiente parte, se solicita lo mismo que en el paso anterior, pero generando data sintética, esta se crea a través de la misma librería _pgmpy_. Primero se lee el dataset y se discretiza igual que en el código anterior, sin embargo esta vez se crea la variable _structureSynt_ la cual contiene el resultado de la creación de la estructura con todo el dataset, con esto se crea _modelSynt_ para el resultado de _.estimate_ del paso anterior y luego, en _modelSyntBayes_ se crea la _DiscreteBayesianNetwork_ y se calculan los parámetros con _.fit_. Se crea _sampler_, el cual es una variable del objeto _BayesianModelSampling_ para poder crear data nueva, y en la variable _synthetic_ se crea nuevos datos con _sampler.forward\_sample(size=50% de dataset original)_.

Una vez obtenida esta nueva data, se aplican algunos cambios a _synthetic_ para que las columnas de este coincidan con las que tiene el dataset y juntan los datos a través de _pd.concat_, se randomiza el orden y se tiene el dataset + data sintética lista. El resto del código a partir de este punto es lo mismo que en la parte uno.