from pandas import read_csv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import random

# Semilla = 100

def indice_saturacion(CS, SS, indice_region, region, indice_fin_entrenamiento, cantidad_dias_entrenamiento):
    Indice = []
    Total_casos_nuevos = []
    # print(indice_fin_entrenamiento)
    for i in range(0, cantidad_dias_entrenamiento): # Suma de casos nuevos con y sin síntomas
        Total_casos_nuevos.append(CS[indice_fin_entrenamiento - i][indice_region + 1] + SS[indice_fin_entrenamiento - i][indice_region + 1])
    
    # print(Total_casos_nuevos)
    for i in range(1, len(Total_casos_nuevos)): # Cálculo del índice de saturación
        dia = Total_casos_nuevos[i]
        if Total_casos_nuevos[i-1] == 0:
            dia_anterior = 1
        else:
            dia_anterior = Total_casos_nuevos[i-1]
        Indice.append(dia / dia_anterior)
    # print(Indice)
    mediana = np.median(Indice)
    # print(Indice)
    # print(Indice[round(len(Indice)/2)])
    # mediana = np.mean(Indice)
    return Indice, mediana

def proyeccion(CS, SS, indice_region, region, cantidad_dias_proyectados, mediana, indice_saturacion, indice_fin_entrenamiento, promedio_regiones):
    # global Semilla
    Casos_nuevos = []
    Casos_reales = []
    Error = []
    Porcentaje = []
    for i in range(0, cantidad_dias_proyectados):
        if len(Casos_nuevos) == 0:
            # random.seed(Semilla)
            if random.random() < 0.7:
                Casos_nuevos.append(round((CS[indice_fin_entrenamiento][indice_region + 1] + SS[indice_fin_entrenamiento][indice_region + 1]) * mediana))
            else:
                # random.seed(Semilla + i)
                aux = random.choice(indice_saturacion)
                while True:
                    if aux > promedio_regiones[indice_region] * 1.2 or aux < promedio_regiones[indice_region] * 0.8:
                        # random.seed(Semilla + i)
                        aux = random.choice(indice_saturacion)
                    else:
                        break
                # print(f"i: {i} - {aux}")
                Casos_nuevos.append(round((CS[indice_fin_entrenamiento][indice_region + 1] + SS[indice_fin_entrenamiento][indice_region + 1]) * aux))
        else:
            # random.seed(Semilla + i)
            if random.random() < 0.7:
                Casos_nuevos.append(round(Casos_nuevos[i-1] * mediana))
            else:
                # random.seed(Semilla + i)
                aux = random.choice(indice_saturacion)
                while True:
                    if aux > promedio_regiones[indice_region] * 1.2 or aux < promedio_regiones[indice_region] * 0.8:
                        # random.seed(Semilla + i)
                        aux = random.choice(indice_saturacion)
                    else:
                        break
                # print(f"i: {i} - {aux}")
                Casos_nuevos.append(round(Casos_nuevos[i-1] * aux))
        Casos_reales.append(CS[indice_fin_entrenamiento + i + 1][indice_region + 1] + SS[indice_fin_entrenamiento + i + 1][indice_region + 1])
    
    for i in range(0, len(Casos_nuevos)):
        Error.append(Casos_nuevos[i] - Casos_reales[i])
        Porcentaje.append(round(abs((Casos_nuevos[i] - Casos_reales[i])) / Casos_reales[i], 3) * 100)
    print("La mediana es: " + str(mediana))
    print("Casos nuevos proyectados:")
    print(Casos_nuevos)
    print("Casos reales:")
    print(Casos_reales)
    print("Diferencia:")
    print(Error)
    print("Porcentaje de error:")
    print(Porcentaje)
    print(f"Promedio de error: {round((np.mean(Porcentaje)), 2)}")


def solucion(CS, SS, indice_region, region, idice_fin_entrenamiento, cantidad_dias_entrenamiento, cantidad_dias_proyectados, promedio_regiones):
    Indice_de_saturacion, mediana = indice_saturacion(CS, SS, indice_region, region, idice_fin_entrenamiento, cantidad_dias_entrenamiento)
    proyeccion(CS, SS, indice_region, region, cantidad_dias_proyectados, mediana, Indice_de_saturacion, idice_fin_entrenamiento, promedio_regiones)



# Obtención de datos de casos nuevos con y sin síntomas por región y fecha, desde el repositorio de datos abiertos del Ministerio de Ciencia
CSintomas = read_csv("https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto26/CasosNuevosConSintomas_T.csv")
Regiones = CSintomas.columns.values[1:17]
CS_out = CSintomas.values[57:] # 57 para que los dos dataset empiecen en la misma fecha

SSintomas = read_csv("https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto27/CasosNuevosSinSintomas_T.csv")
SS_out = SSintomas.values

Cantidad_dias_proyectados = 10 # Cantidad de días a proyectar
Cantidad_dias_entrenamiento = 65 # Cantidad de días a utilizar para entrenar el modelo

Fecha_inicio_dataset = datetime(2020, 4, 29).date() # Fecha de inicio del dataset
Fecha_fin_entrenamiento = datetime(2022, 10, 12).date() # Fecha de termino de entrenamiento
# print(Fecha_fin_entrenamiento - timedelta(days=Cantidad_dias_entrenamiento))
# Fecha_fin_entrenamiento = Fecha_fin_entrenamiento + timedelta(days=Cantidad_dias_entrenamiento) # Fecha de fin de entrenamiento
Indice_fecha_fin_entrenamiento = (Fecha_fin_entrenamiento - Fecha_inicio_dataset).days # Indice de la fecha de inicio de entrenamiento en el dataset
# print(CS_out[Indice_fecha_fin_entrenamiento - 1][0]) # Fecha de inicio de entrenamiento

promedio_regiones = []

for i in range(0, len(Regiones)):
    Indice = []
    Total_casos_nuevos = []
    for j in range(0, Indice_fecha_fin_entrenamiento): # Suma de casos nuevos con y sin síntomas
        Total_casos_nuevos.append(CS_out[j][i+1] + SS_out[j][i+1])
        
    for k in range(0, len(Total_casos_nuevos)): # Cálculo del índice de saturación
        dia = Total_casos_nuevos[k]
        if Total_casos_nuevos[k-1] == 0:
            dia_anterior = 1
        else:
            dia_anterior = Total_casos_nuevos[k-1]
        Indice.append(dia / dia_anterior)
    promedio_regiones.append(np.mean(Indice))

# for l in range(0, len(Regiones)):
#     print(Regiones[l], promedio_regiones[l])

for p in range(0, len(Regiones)):
    print(f"region: {p}")
    solucion(CS_out, SS_out, p, Regiones, Indice_fecha_fin_entrenamiento, Cantidad_dias_entrenamiento, Cantidad_dias_proyectados, promedio_regiones)