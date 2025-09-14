import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Posiciones_Escalera = {2: 11, 6: 24, 13: 43, 16: 37, 19: 30, 40: 50}
Posiciones_Serpiente = {49: 17, 46: 25, 41: 22, 36: 15, 23: 11, 18: 10}
tamaño_tablero = 50

def construir_matriz_transicion():
    matriz_transicion = np.zeros((tamaño_tablero, tamaño_tablero))
    
    for indice_casilla in range(tamaño_tablero - 1):
        casilla_actual = indice_casilla + 1
        

        if casilla_actual in Posiciones_Escalera:
            casilla_destino = Posiciones_Escalera[casilla_actual]

            if casilla_destino == tamaño_tablero:
                casilla_destino = 1
            matriz_transicion[indice_casilla, casilla_destino - 1] = 1.0
        

        elif casilla_actual in Posiciones_Serpiente:
            casilla_destino = Posiciones_Serpiente[casilla_actual]
 
            if casilla_destino == tamaño_tablero:
                casilla_destino = 1
            matriz_transicion[indice_casilla, casilla_destino - 1] = 1.0
        

        else:
            for resultado_dado in range(1, 7):
                proxima_casilla = casilla_actual + resultado_dado
                
                if proxima_casilla > tamaño_tablero:
                    siguiente_indice = indice_casilla 
                elif proxima_casilla in Posiciones_Escalera:
                    destino = Posiciones_Escalera[proxima_casilla]

                    if destino == tamaño_tablero:
                        destino = 1
                    siguiente_indice = destino - 1
                elif proxima_casilla in Posiciones_Serpiente:
                    destino = Posiciones_Serpiente[proxima_casilla]

                    if destino == tamaño_tablero:
                        destino = 1
                    siguiente_indice = destino - 1
                else:
                    siguiente_indice = proxima_casilla - 1

                    if siguiente_indice == tamaño_tablero - 1:
                        siguiente_indice = 0
                
                matriz_transicion[indice_casilla, siguiente_indice] += 1/6

    matriz_transicion[tamaño_tablero - 1, 0] = 1.0
    
    return matriz_transicion
 

def jugar_Random_walk(Cantidad_de_partidas):
    Matriz_juego = np.zeros(tamaño_tablero)
    
    for partida in range(Cantidad_de_partidas):
        Posicion = 0  
        
        while Posicion < tamaño_tablero - 1:
            tirada = np.random.randint(1, 7)
            
            if tirada == 6:  # Repetir turno si sale 6
                continue
            
            Ubicacion_proxima = Posicion + tirada
            if Ubicacion_proxima >= tamaño_tablero:
                Ubicacion_proxima = Posicion  # Para que no vaya a una posicion mayor al 50
            

            casilla_real = Ubicacion_proxima + 1  
            if casilla_real in Posiciones_Escalera:
                Ubicacion_proxima = Posiciones_Escalera[casilla_real] - 1
            elif casilla_real in Posiciones_Serpiente:
                Ubicacion_proxima = Posiciones_Serpiente[casilla_real] - 1
            

            Matriz_juego[Ubicacion_proxima] += 1 
            Posicion = Ubicacion_proxima
        
        Matriz_juego[49] = 0 #Se pone para que 50 sea 0, ya que va a 1 directamente
        Matriz_juego[0] += 1 #Una vez llega a 50 pasa a 1 directamente 
    
    pi = Matriz_juego / Matriz_juego.sum()#Nomaliza los resultados 
    
    # Guardar en CSV
    df = pd.DataFrame(pi, columns=['Probabilidad'])
    df.index += 1  # Casillas 1–50
    df.to_csv("matriz_pi.csv", index_label="Casilla")
    
    return pi

def metodo_exacto(Matriz_T):
     #Como calcularlo (Mediante la ecuacion pi = (T-pi) -> 0 = pi(T-I)) Numpy calcula sistemas lineales con vectores columnares 
    Matriz_Exacta = Matriz_T.T - np.eye(tamaño_tablero) #Transpone la matriz y crea matriz identidad : 0 = pi(T-I)
    Matriz_Exacta[-1,:] = np.ones(tamaño_tablero) #Se crea para la condicion que la sumatoria de todas la probabilidades deben dar 1
    Matriz_Final = np.zeros(tamaño_tablero)
    Matriz_Final[-1] = 1
    pi = np.linalg.solve(Matriz_Exacta,Matriz_Final)
    pi = np.where(pi < 1e-3, 0, pi)#Se eliminan los valores tan pequeños que no aportan nada
    return pi

def metodo_iterativo(Matriz_T, Cantidad_de_ejecuciones):
    tolerancia = 1e-12 #Este define  el valor en que punto ya la distribucion ya no cambia para detener la iteracion 
    Matriz_pi = np.ones(tamaño_tablero) / tamaño_tablero  # Distribucion uniforme yaque el dado no tiene ninguna carga 
    
    # Aquí guardamos TODA la evolución de pi
    historia = [Matriz_pi.copy()]

    for i in range(Cantidad_de_ejecuciones):
        Matriz_Proxima = Matriz_pi @ Matriz_T  # El @ Es para hacer el producto entre 2 matrices  donde se hace la matriz (PI) x La de transicion iterando n veces

        if np.linalg.norm(Matriz_Proxima - Matriz_pi, 1) < tolerancia: #Este define en que punto ya la distribucion no cambia y seguir iterando ya no vale la pena 
            Matriz_pi = Matriz_Proxima
            historia.append(Matriz_pi.copy())
            break

        Matriz_pi = Matriz_Proxima
        historia.append(Matriz_pi.copy())
    
    # Limpieza
    Limpieza = 1e-11  #Este eliminara los valores muy pequeños que no vale la pena tenerlos en la matriz 
    Matriz_pi[Matriz_pi < Limpieza] = 0
    Matriz_pi = Matriz_pi / Matriz_pi.sum()
    
    # Convertimos la historia en DataFrame para exportar
    df_historia = pd.DataFrame(historia)
    df_historia.index.name = "Iteración"
    df_historia.columns = [f"Casilla_{i+1}" for i in range(tamaño_tablero)]
    df_historia.to_csv("convergencia_variables.csv")

    # Gráfico: cada variable en su propia curva
    plt.figure(figsize=(10,6))
    for col in df_historia.columns:
        plt.plot(df_historia.index, df_historia[col], alpha=0.7)
    plt.title("Convergencia de cada variable π(i)")
    plt.xlabel("Iteración")
    plt.ylabel("Probabilidad")
    plt.grid(True)
    plt.savefig("convergencia_variables.png", dpi=300)
    plt.close()
    
    return Matriz_pi, df_historia

def duracion_esperada(Cantidad_de_partidas, graficar=False):
    duraciones = []
    for i in range(Cantidad_de_partidas):
        Posicion = 0
        turnos = 0
        while Posicion < tamaño_tablero - 1:
            tirada = np.random.randint(1, 7)
            
            if tirada == 6:
                continue
            
            Ubicacion_proxima = Posicion + tirada
            
            if Ubicacion_proxima >= tamaño_tablero:
                Ubicacion_proxima = Posicion
            casilla_real = Ubicacion_proxima + 1
            
            if casilla_real in Posiciones_Escalera:
                Ubicacion_proxima = Posiciones_Escalera[casilla_real] - 1
                
            elif casilla_real in Posiciones_Serpiente:
                Ubicacion_proxima = Posiciones_Serpiente[casilla_real] - 1
                
            Posicion = Ubicacion_proxima
            turnos += 1
        duraciones.append(turnos)

    if graficar:
        plt.hist(duraciones, bins=50, density=True)
        plt.xlabel("Turnos")
        plt.ylabel("Frecuencia")
        plt.title("Duración de las partidas")
        plt.savefig("Duracion_partidas.png", dpi=300, bbox_inches = "tight")
        plt.show()
    

    return np.mean(duraciones)


def vector_visitas(Cantidad_de_partidas, graficar=True,guardar = True):
    Matriz_visitas = np.zeros(tamaño_tablero)
    
    for i in range(Cantidad_de_partidas):
        Posicion = 0
        while True:
            tirada = np.random.randint(1, 7)
            if tirada == 6:
                continue

            Ubicacion_proxima = Posicion + tirada
            if Ubicacion_proxima >= tamaño_tablero:
                Ubicacion_proxima = Posicion

            casilla_real = Ubicacion_proxima + 1
            if casilla_real in Posiciones_Escalera:
                Ubicacion_proxima = Posiciones_Escalera[casilla_real] - 1
            elif casilla_real in Posiciones_Serpiente:
                Ubicacion_proxima = Posiciones_Serpiente[casilla_real] - 1


            if Ubicacion_proxima == tamaño_tablero - 1:
                Matriz_visitas[0] += 1
                Posicion = 0
                break
            else:
                Matriz_visitas[Ubicacion_proxima] += 1
                Posicion = Ubicacion_proxima

    Vector_promedio = Matriz_visitas / Cantidad_de_partidas

    if graficar:
        plt.figure(figsize=(12,6))
        plt.bar(range(1, tamaño_tablero + 1), Vector_promedio, color='skyblue', edgecolor='black')
        plt.xlabel("Casilla")
        plt.ylabel("Visitas promedio por partida")
        plt.title("promedio de visitas por partida")
        plt.xticks(range(1, tamaño_tablero + 1))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig("Promedio_caidas_por_partidas.png", dpi=300, bbox_inches = "tight")
        plt.show()
    

    return Vector_promedio


Matriz_t = construir_matriz_transicion()
pd.DataFrame(Matriz_t).to_csv("matriz_transicion.csv", index=False)

pi_exacta = metodo_exacto(Matriz_t)

df_pi_exacta = pd.DataFrame(pi_exacta, columns=["Probabilidad"])
df_pi_exacta.index += 1
df_pi_exacta.to_csv("pi_exacta.csv", index_label="Casilla")

Matriz_P_iterativa, Historia_de_convergencia = metodo_iterativo(Matriz_t, Cantidad_de_ejecuciones=20000)

df_pi_iter = pd.DataFrame(Matriz_P_iterativa, columns=["Probabilidad"])
df_pi_iter.index += 1
df_pi_iter.to_csv("pi_iterativo.csv", index_label="Casilla")

Matriz_Pi_Jugando = jugar_Random_walk(100000)  

Duracion_media = duracion_esperada(100000, graficar=True)

print(f"Duración promedio de la partida: {Duracion_media:.2f} turnos")

Vector_de_visitas = vector_visitas(100000, graficar=True)

df_visitas = pd.DataFrame(Vector_de_visitas, columns=["Visitas_promedio"])
df_visitas.index += 1
df_visitas.to_csv("vector_visitas.csv", index_label="Casilla")
