import pandas as pd

#   Leer csv
dataset = pd.read_csv("dataset/vgsales.csv")
#   Eliminar columnas que no sirven para bayes
dataset = dataset.drop(["Rank", "Name"], axis = 1)
#   Discretizar datos numericos
dataset["Year"] = pd.cut(dataset["Year"], bins=[0, 1990, 2000, 2010, 2020], 
                         labels=["<1990", "1990-2000", "2000-2010", "2010-2020"])
dataset["NA_Sales"] = pd.cut(dataset["NA_Sales"], bins=3,
                              labels=["Bajo", "Medio", "Alto"])
dataset["EU_Sales"] = pd.cut(dataset["EU_Sales"], bins=3,
                              labels=["Bajo", "Medio", "Alto"])
dataset["JP_Sales"] = pd.cut(dataset["JP_Sales"], bins=3,
                              labels=["Bajo", "Medio", "Alto"])
dataset["Other_Sales"] = pd.cut(dataset["Other_Sales"], bins=3,
                              labels=["Bajo", "Medio", "Alto"])
dataset["Global_Sales"] = pd.cut(dataset["Global_Sales"], bins=3,
                              labels=["Bajo", "Medio", "Alto"])
#   Separación de datos
dataset70 = dataset.sample(frac=0.7, random_state=42)
dataset30 = dataset.drop(dataset70.index)