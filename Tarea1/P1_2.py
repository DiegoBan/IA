import pandas as pd
from pgmpy.estimators import HillClimbSearch, BIC, BDeu, BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
import io, contextlib, logging
logging.getLogger("pgmpy").setLevel(logging.WARNING)
import json
from tqdm import tqdm

#   Leer csv
dataset = pd.read_csv("dataset/vgsales.csv")
#   Eliminar columnas que no sirven para bayes
dataset = dataset.drop(["Rank", "Name"], axis = 1).dropna()
#   Discretizar datos numericos
dataset["Year"] = pd.cut(dataset["Year"], bins=[0, 1990, 2000, 2010, 2020], 
                         labels=["<1990", "1990-2000", "2000-2010", "2010-2020"])
dataset["NA_Sales"] = pd.cut(dataset["NA_Sales"], bins=3,
                              labels=["Low", "Mid", "High"])
dataset["EU_Sales"] = pd.cut(dataset["EU_Sales"], bins=3,
                              labels=["Low", "Mid", "High"])
dataset["JP_Sales"] = pd.cut(dataset["JP_Sales"], bins=3,
                              labels=["Low", "Mid", "High"])
dataset["Other_Sales"] = pd.cut(dataset["Other_Sales"], bins=3,
                              labels=["Low", "Mid", "High"])
dataset["Global_Sales"] = pd.cut(dataset["Global_Sales"], bins=3,
                              labels=["Low", "Mid", "High"])
#   Generación de modelo para crear data sintetica
structureSynt = HillClimbSearch(dataset)
modelSynt = structureSynt.estimate(scoring_method=BDeu(dataset))
print("\n========== Aristas Aprendidas (modelSynt) ==========\n", list(modelSynt.edges()))
modelSyntBayes = DiscreteBayesianNetwork(modelSynt.edges())
modelSyntBayes.fit(dataset, estimator=BayesianEstimator)
#   Generación de data sintetica
syntDataLen = int(len(dataset)*0.5)
sampler = BayesianModelSampling(modelSyntBayes)
synthetic = sampler.forward_sample(size=syntDataLen)
print("Synthetic shape: ", synthetic.shape)
print(synthetic.head())