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
print("Data sintetica generada: ", synthetic.shape)
#   Concatenación de dataset con nueva sintetica
data_cols = list(dataset.columns)
synthetic_cols = list(synthetic.columns)
synthetic = synthetic[data_cols]    # Reordena columnas en el orden de dataset
new_dataset = pd.concat([dataset, synthetic], ignore_index=True)
new_dataset = new_dataset.sample(frac=1, random_state=42).reset_index(drop=True)    # Aleatoriza orden de filas
print("Nuevo tamaño de datos (dataset + sinteticos): ", len(new_dataset))


#   Separación de datos
dataset70 = dataset.sample(frac=0.7, random_state=42)
dataset30 = dataset.drop(dataset70.index)
print("Dataset discretizado y separado")
print(dataset.info())
#   Aprendizaje de estrucutra: HillClimbingSearch
structure = HillClimbSearch(dataset70)
model_BIC = structure.estimate(scoring_method=BIC(dataset70))
print("\n========== Aristas Aprendidas (model1 scoring_method=BIC) ==========\n", list(model_BIC.edges()))

model_BDeu = structure.estimate(scoring_method=BDeu(dataset70))
print("\n========== Aristas Aprendidas (model2 scoring_method=BDeu) ==========\n", list(model_BDeu.edges()))

model1 = DiscreteBayesianNetwork(model_BIC.edges())
model2 = DiscreteBayesianNetwork(model_BDeu.edges())
#   Estimación de parámetros:
model1.fit(dataset70, estimator=BayesianEstimator)
print("\n========== Estimaciones de model1 (scoring_method=BIC) ==========\n")
for cpd in model1.get_cpds():
    print(cpd)

model2.fit(dataset70, estimator=BayesianEstimator)
print("\n========== Estimaciones de model2 (scoring_method=BDeu) ==========\n")
for cpd in model2.get_cpds():
    print(cpd)

#   Inferencias
print("\n========== Inferencias en model1 (scoring_method=BIC) ==========\n")
infer1 = VariableElimination(model1)
print("P(Genre | Platform=Wii):\n", 
      infer1.query(variables=["Genre"], evidence={"Platform": "Wii"}))
print("P(Year | Platform=NES):\n",
      infer1.query(variables=["Year"], evidence={"Platform": "NES"}))
print("P(NA_Sales | Global_Sales=High):\n",
      infer1.query(variables=["NA_Sales"],evidence={"Global_Sales": "High"}))
print("P(Global_Sales | EU_Sales=Mid):\n",
      infer1.query(variables=["Global_Sales"],evidence={"EU_Sales": "Mid"}))

print("\n========== Inferencias en model2 (scoring_method=BDeu) ==========\n")
infer2 = VariableElimination(model2)
print("P(Genre | Platform=Wii):\n",
      infer2.query(variables=["Genre"], evidence={"Platform": "Wii"}))
print("P(Platform | Year=2000-2010, Genre=Sports):\n",
      infer2.query(variables=["Platform"], evidence={"Year": "2000-2010", "Genre": "Sports"}))
print("P(Publisher | Platform=PS4, JP_Sales=Low):\n",
      infer2.query(variables=["Publisher"], evidence={"Platform": "PS4", "JP_Sales": "Low"}))
print("P(NA_Sales, EU_Sales | Global_Sales= High):\n",
      infer2.query(variables=["NA_Sales", "EU_Sales"], evidence={"Global_Sales": "High"}))

#   Validación
qinf = pd.read_csv("dataset/queries_inferences.csv")
total_inferences = len(qinf) * len(dataset30)

progress = tqdm(total=total_inferences, desc="Validacion 1")
total_cases1 = 0
fav_cases1 = 0
for _, row in qinf.iterrows():
    target = row['target']
    evidences_cols = json.loads(row['evidence'])
    for _, rowD in dataset30.iterrows():
        evidences = {}
        for col in evidences_cols:
            evidences[col] = rowD[col]
        #   Inferencia
        _buf_out = io.StringIO()
        _buf_err = io.StringIO()
        try:
            with contextlib.redirect_stdout(_buf_out), contextlib.redirect_stderr(_buf_err):
                resp = infer1.map_query(variables=[target], evidence=evidences)
        except Exception as e:
            captured: _buf_out.getvalue() + _buf_err.getvalue()
            print("Error during inference:", e)
            if captured:
                print("Captured output:\n", captured)
            raise
        #   Comparación con dato real
        pred = resp.get(target) if isinstance(resp, dict) else resp
        if pred == rowD[target]:
            fav_cases1 += 1
        total_cases1 += 1
        progress.update(1)
progress.close()

progress = tqdm(total=total_inferences, desc="Validacion 2")
total_cases2 = 0
fav_cases2 = 0
for _, row in qinf.iterrows():
    target = row['target']
    evidences_cols = json.loads(row['evidence'])
    for _, rowD in dataset30.iterrows():
        evidences = {}
        for col in evidences_cols:
            evidences[col] = rowD[col]
        #   Inferencia
        _buf_out = io.StringIO()
        _buf_err = io.StringIO()
        try:
            with contextlib.redirect_stdout(_buf_out), contextlib.redirect_stderr(_buf_err):
                resp = infer2.map_query(variables=[target], evidence=evidences)
        except Exception as e:
            captured: _buf_out.getvalue() + _buf_err.getvalue()
            print("Error during inference:", e)
            if captured:
                print("Captured output:\n", captured)
            raise
        #   Comparación con dato real
        pred = resp.get(target) if isinstance(resp, dict) else resp
        if pred == rowD[target]:
            fav_cases2 += 1
        total_cases2 += 1
        progress.update(1)
progress.close()

porc_model1 = (fav_cases1/total_cases1)*100
porc_model2 = (fav_cases2/total_cases2)*100
print("Validacion modelo 1: ", porc_model1, "porcieto de casos acertados")
print("Validacion modelo 2: ", porc_model2, "porcieto de casos acertados")
print("Diferencia entre modelo 1 y 2: ", abs(((fav_cases1/total_cases1)*100)-((fav_cases2/total_cases2)*100)))
if porc_model1 > porc_model2:
    print("Mejor modelo: Model 1")
else:
    print("Mejor modelo: Model 2")