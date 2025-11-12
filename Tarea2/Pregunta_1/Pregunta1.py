from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score, pairwise_distances_argmin
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances_argmin
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter

def Ejecucion_K_means(X,Cantidad_de_Clusters,Semilla_de_generacion,N_de_ejecuciones,N_de_Iteraciones) : 
    #class sklearn.cluster.KMeans(n_clusters=8, *, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')
    Ejecucion = KMeans(n_clusters=Cantidad_de_Clusters, random_state=Semilla_de_generacion, init='random', n_init=N_de_ejecuciones,max_iter=N_de_Iteraciones)# fit({N_Muestras,N_caracteristicas},ignorar,Peso(Ignorar))
    Etiquetas_de_la_ejecucion = Ejecucion.fit_predict(X) #Etiquetas de cada punto
    
    Puntuacion= silhouette_score(X, Etiquetas_de_la_ejecucion)
    Puntuacion_calinski = calinski_harabasz_score(X, Etiquetas_de_la_ejecucion)
    
    return Ejecucion, Etiquetas_de_la_ejecucion, Puntuacion, Puntuacion_calinski

def Ejecucion_K_meansplusplus(X,Cantidad_de_Clusters,Semilla_de_generacion,N_de_ejecuciones,N_de_Iteraciones) : 
    #class sklearn.cluster.KMeans(n_clusters=8, *, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')
    Ejecucion = KMeans(n_clusters=Cantidad_de_Clusters, random_state=Semilla_de_generacion, init='k-means++',n_init=N_de_ejecuciones,max_iter=N_de_Iteraciones)# fit({N_Muestras,N_caracteristicas},ignorar,Peso(Ignorar))
    Etiquetas_de_la_ejecucion = Ejecucion.fit_predict(X) #Etiquetas de cada punto
    
    Puntuacion_silhouette= silhouette_score(X, Etiquetas_de_la_ejecucion)
    Puntuacion_calinski = calinski_harabasz_score(X, Etiquetas_de_la_ejecucion)
    
    return Ejecucion,  Etiquetas_de_la_ejecucion , Puntuacion_silhouette , Puntuacion_calinski

def Ejecucion_MeanShift(X,Percentil,Numero_de_muestras,Semilla_de_generacion) : 
    #class sklearn.cluster.MeanShift(*, bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=None, max_iter=300)
    #sklearn.cluster.estimate_bandwidth(X, *, quantile=0.3, n_samples=None, random_state=0, n_jobs=None)[source]
    #Quantile sirve para controlar el tamaño del radio
    Bandwidth_Estimado = estimate_bandwidth(X, quantile=Percentil, n_samples=Numero_de_muestras, random_state=Semilla_de_generacion)
    Ejecucion = MeanShift(bandwidth=Bandwidth_Estimado, bin_seeding=True, min_bin_freq=1, cluster_all=True, n_jobs=None, max_iter=300)
    Etiquetas_de_la_ejecucion = Ejecucion.fit_predict(X)
    
    if len(set(Etiquetas_de_la_ejecucion)) > 1 :
        
        Puntuacion_silhouette= silhouette_score(X, Etiquetas_de_la_ejecucion)
        
        Puntuacion_calinski = calinski_harabasz_score(X, Etiquetas_de_la_ejecucion)
        
    else : 
        
        return Ejecucion, Etiquetas_de_la_ejecucion, None, None
    
    return Ejecucion,  Etiquetas_de_la_ejecucion , Puntuacion_silhouette , Puntuacion_calinski

def Obtener_mejor(resultados) :
    mejor = None
    if resultados is None :
        print("No hay datos para trabajar")
        return None
    
    for i in resultados :
        if i["Silhouette"] is not None :
           if mejor is None or mejor["Silhouette"] < i["Silhouette"] :
                mejor = i
        
    return mejor

def Evaluar_Clusters(etiquetas, etiquetas_reales):
    mapeo = {}
    for cluster in np.unique(etiquetas):
        indices = np.where(etiquetas == cluster)
        clases_en_cluster = etiquetas_reales.iloc[indices]
        clase_dominante = Counter(clases_en_cluster).most_common(1)[0][0]
        mapeo[cluster] = clase_dominante

    etiquetas_mapeadas = [mapeo[c] for c in etiquetas]
    Fiabilidad = np.mean(np.array(etiquetas_mapeadas) == np.array(etiquetas_reales))
    
    return Fiabilidad, mapeo


datos = pd.read_csv(r"/home/doshuertos/Escritorio/IA/Tarea2/csv/filtered_dataset.csv")
Y = datos["diabetes_stage"]
X = datos.drop(columns=["diabetes_stage"])
Datos_Entrenamiento, Datos_Prueba, Etiquetas_Entrenamiento, Etiquetas_Prueba = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

resultados_Kmeans = []
resultados_Kmeansplusplus = []
resultados_meanshift = []

puntuacion = 0;
Pruebas_Kmeans = [
    {"Cantidad_de_Clusters": 3, "Semilla_de_generacion": 42, "N_de_ejecuciones": 10, "N_de_Iteraciones": 100},
    {"Cantidad_de_Clusters": 4, "Semilla_de_generacion": 7,  "N_de_ejecuciones": 15, "N_de_Iteraciones": 200},
    {"Cantidad_de_Clusters": 5, "Semilla_de_generacion": 21, "N_de_ejecuciones": 5,  "N_de_Iteraciones": 150},
    {"Cantidad_de_Clusters": 6, "Semilla_de_generacion": 99, "N_de_ejecuciones": 10, "N_de_Iteraciones": 250},
]

for i, pruebas in enumerate(Pruebas_Kmeans, 1):
    modelo, etiquetas, sil, cal = Ejecucion_K_means(Datos_Entrenamiento, **pruebas)
    print(f"Datos de la {i}: {pruebas}, (Kmeans)")
    print(f"Clusters Generados: {len(set(etiquetas))}")
    print(f"puntuacion Silhouette : {sil:.4f}")
    print(f"puntuacion Calinski-Harabasz : {cal:.2f}")
    resultados_Kmeans.append({"Config": pruebas, "Modelo": modelo, "Etiquetas": etiquetas, "Silhouette": sil, "Calinski": cal})
    print("\n")
    
        

Pruebas_Kmeansplusplus = [
    {"Cantidad_de_Clusters": 3, "Semilla_de_generacion": 42, "N_de_ejecuciones": 10, "N_de_Iteraciones": 100},
    {"Cantidad_de_Clusters": 4, "Semilla_de_generacion": 7,  "N_de_ejecuciones": 15, "N_de_Iteraciones": 200},
    {"Cantidad_de_Clusters": 5, "Semilla_de_generacion": 21, "N_de_ejecuciones": 5,  "N_de_Iteraciones": 150},
    {"Cantidad_de_Clusters": 6, "Semilla_de_generacion": 99, "N_de_ejecuciones": 10, "N_de_Iteraciones": 250},
]

for i, pruebas in enumerate(Pruebas_Kmeansplusplus, 1):
    modelo, etiquetas, sil, cal = Ejecucion_K_meansplusplus(Datos_Entrenamiento, **pruebas)
    print(f"Datos de la {i}: {pruebas}, (Kmeans++)")
    print(f"Clusters Generados: {len(set(etiquetas))}")
    print(f"puntuacion Silhouette : {sil:.4f}")
    print(f"puntuacion Calinski-Harabasz : {cal:.2f}")
    resultados_Kmeansplusplus.append({"Config": pruebas, "Modelo": modelo, "Etiquetas": etiquetas, "Silhouette": sil, "Calinski": cal})
    print("\n")

tests_meanshift = [
    {"Percentil": 0.01, "Numero_de_muestras": 500, "Semilla_de_generacion": 42},
    {"Percentil": 0.05, "Numero_de_muestras": 500, "Semilla_de_generacion": 42},
    {"Percentil": 0.1, "Numero_de_muestras": 500, "Semilla_de_generacion": 42},
    {"Percentil": 0.2, "Numero_de_muestras": 500, "Semilla_de_generacion": 42},
]

for i, pruebas in enumerate(tests_meanshift, 1):
    modelo, etiquetas, sil, cal = Ejecucion_MeanShift(Datos_Entrenamiento, **pruebas)
    n_clusters = len(set(etiquetas))
    print(f"Datos de la {i}: {pruebas}, (Meanshift)")
    print(f"Clusters encontrados: {n_clusters}")
    if sil is not None:
        print(f"puntuacion Silhouette : {sil:.4f}")
        print(f"puntuacion Calinski-Harabasz : {cal:.2f}")
        resultados_meanshift.append({"Config": pruebas, "Modelo": modelo, "Etiquetas": etiquetas, "Silhouette": sil, "Calinski": cal})
    else:
        print("Solo se creo 1 cluster, las medidas utilizadan no fueron de la mejor magnitud")
    print("\n")

Mejor_kmeans = Obtener_mejor(resultados_Kmeans)
Mejor_kmeansplusplus = Obtener_mejor(resultados_Kmeansplusplus)
Mejor_meanshif = Obtener_mejor(resultados_meanshift)


for nombre, mejor in zip(["K-Means", "K-Means++", "MeanShift"], [Mejor_kmeans, Mejor_kmeansplusplus, Mejor_meanshif]):
    print(f"Resultados de {nombre}")
    if mejor is not None:
        print(f"Puntacion Silhouette : {mejor['Silhouette']:.4f}")
        print(f"Puntacion Calinski-Harabasz: {mejor['Calinski']:.2f}")
        print(f"Datos Utilizados: {mejor['Config']}")
        print(f"Clusters Totales: {len(set(mejor['Etiquetas']))}")
    else:
        print("No se encontró un resultado válido.")


modelo_Kmeans = Mejor_kmeans["Modelo"]
Prueba_de_etiquetas_Kmeans = modelo_Kmeans.predict(Datos_Prueba)
Fiabilidad_kmeans, mapeo_kmeans = Evaluar_Clusters(Prueba_de_etiquetas_Kmeans, Etiquetas_Prueba)
print(f"Fiabilidad de K-Means (comparando con Y real): {Fiabilidad_kmeans:.4f}")
print(f"mapa de las Etiquetas: {mapeo_kmeans}\n")

modelo_kmeansplusplus = Mejor_kmeansplusplus["Modelo"]
Prueba_de_etiquetas_kmeansplusplus = modelo_kmeansplusplus.predict(Datos_Prueba)
Fiabilidad_kmeanspp, mapeo_kmeanspp = Evaluar_Clusters(Prueba_de_etiquetas_kmeansplusplus, Etiquetas_Prueba)
print(f"Fiabilidad de K-Means++: {Fiabilidad_kmeanspp:.4f}")
print(f"mapa de las Etiquetas: {mapeo_kmeanspp}\n")

modelo_meanshift = Mejor_meanshif["Modelo"]
Prueba_de_etiquetas_meanshift = pairwise_distances_argmin(Datos_Prueba, modelo_meanshift.cluster_centers_)
Fiabilidad_meanshift, mapeo_meanshift = Evaluar_Clusters(Prueba_de_etiquetas_meanshift, Etiquetas_Prueba)
print(f"Fiabilidad del MeanShift: {Fiabilidad_meanshift:.4f}")
print(f"mapa de las Etiquetas: {mapeo_meanshift}\n")



def graficar_clusters(X, etiquetas_cluster, etiquetas_reales, titulo):
    pca = PCA(n_components=2)
    X_reducido = pca.fit_transform(X)
    
    plt.figure(figsize=(7,5))
    scatter = plt.scatter(X_reducido[:,0], X_reducido[:,1], c=etiquetas_cluster, cmap='viridis', alpha=0.6)
    plt.title(f"{titulo} - Clusters formados")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.colorbar(scatter, label="Cluster")
    plt.show()
    
    plt.figure(figsize=(7,5))
    scatter = plt.scatter(X_reducido[:,0], X_reducido[:,1], c=etiquetas_reales.astype('category').cat.codes, cmap='plasma', alpha=0.6)
    plt.title(f"{titulo} - Etiquetas reales (Y)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.colorbar(scatter, label="Etiqueta real")
    plt.show()

graficar_clusters(Datos_Prueba, Prueba_de_etiquetas_Kmeans, Etiquetas_Prueba, "K-Means")

graficar_clusters(Datos_Prueba, Prueba_de_etiquetas_kmeansplusplus, Etiquetas_Prueba, "K-Means++")
graficar_clusters(Datos_Prueba, Prueba_de_etiquetas_meanshift, Etiquetas_Prueba, "MeanShift")