import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BIC, BDeu, BayesianEstimator
import logging
logging.getLogger("pgmpy").setLevel(logging.WARNING)

#   Leer csv
dataset = pd.read_csv("dataset/vgsales.csv")
#   Eliminar columnas que no sirven para bayes
dataset = dataset.drop(["Rank", "Name"], axis = 1).dropna()
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
dataset70 = dataset.sample(frac=0.7, random_state=42)   # Toma 70% de los datos aleatorio con seed 42
dataset30 = dataset.drop(dataset70.index)               # Toma el 30% de datos restante
print("Dataset discretizado y separado")
print(dataset.info())
#   Aprendizaje de estrucutra: HillClimbingSearch
structure = HillClimbSearch(dataset70)
model_BIC = structure.estimate(scoring_method=BIC(dataset70))
print("\n===== Aristas Aprendidas (model1 scoring_method=BIC) =====\n", list(model_BIC.edges()))

model_BDeu = structure.estimate(scoring_method=BDeu(dataset70))
print("\n===== Aristas Aprendidas (model2 scoring_method=BDeu) =====\n", list(model_BDeu.edges()))

model1 = DiscreteBayesianNetwork(model_BIC.edges())
model2 = DiscreteBayesianNetwork(model_BDeu.edges())
#   Estimación de parámetros:
model1.fit(dataset70, estimator=BayesianEstimator)
print("========== Estimaciones de model1 (scoring_method=BIC) ==========")
for cpd in model1.get_cpds():
    print(cpd)

model2.fit(dataset70, estimator=BayesianEstimator)
print("========== Estimaciones de model1 (scoring_method=BDeu) ==========")
for cpd in model2.get_cpds():
    print(cpd)