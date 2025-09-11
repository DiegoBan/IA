# Tarea 1 - Inteligencia Artificial

En esta carpeta se desarrolla la tarea N°1 de Inteligencia Artificial, la cual consta de dos partes principales, a través de este README se explicará todo lo necesario para cada una de estas mismas y el resultado final almacenado en un Google Collab.

### Ejecución y utilización de python

Para ejecutar esto de una manera controlada y sin crear posibles errores por dependencias con otras librerías o funciones de python en nuestro computador (local), se utilizará un entorno virtual de python "virtualenv". Para la utilización de este entorno se tendrán los siguientes comandos de referencia:

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

