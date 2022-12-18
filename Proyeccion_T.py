from pandas import read_csv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import random

Semilla = 7
def CalculoCamas(arreglo_camas,casos,indice,region,indice_x):
    camasCovid=arreglo_camas[region+17][indice]
    camas=arreglo_camas[region][indice]
    camasTotales=camas
    x=casos[:]
    while True:
        camasTotales=camasTotales
        casos=x[indice_x]
        aux=((camasCovid*casos)/((1-(camasCovid/camasTotales))*camasTotales))
        if 0.9>= aux:
            break
        camasTotales=camasTotales+1

    return aux, camasTotales

def sat(arreglo_camas,casos,indice,region,indice_x):
    camasCovid=arreglo_camas[region+17][indice]
    camas=arreglo_camas[region][indice]
    casos=x[indice_x]
    saturacion=((camasCovid*casos)/((1-(camasCovid/camas))*camas))

    return saturacion


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
    global Semilla
    Casos_nuevos = []
    Casos_reales = []
    Error = []
    Porcentaje = []
    for i in range(0, cantidad_dias_proyectados):
        if len(Casos_nuevos) == 0:
            Semilla += 1
            random.seed(Semilla)
            if random.random() < 0.7:
                Casos_nuevos.append(round((CS[indice_fin_entrenamiento][indice_region + 1] + SS[indice_fin_entrenamiento][indice_region + 1]) * mediana))
            else:
                Semilla += 1
                random.seed(Semilla)
                aux = random.choice(indice_saturacion)
                while True:
                    if aux > promedio_regiones[indice_region] * 1.2 or aux < promedio_regiones[indice_region] * 0.8:
                        Semilla += 1
                        random.seed(Semilla)
                        aux = random.choice(indice_saturacion)
                    else:
                        break
                # print(f"i: {i} - {aux}")
                Casos_nuevos.append(round((CS[indice_fin_entrenamiento][indice_region + 1] + SS[indice_fin_entrenamiento][indice_region + 1]) * aux))
        else:
            Semilla += 1
            random.seed(Semilla)
            if random.random() < 0.7:
                Casos_nuevos.append(round(Casos_nuevos[i-1] * mediana))
            else:
                Semilla += 1
                random.seed(Semilla)
                aux = random.choice(indice_saturacion)
                while True:
                    if aux > promedio_regiones[indice_region] * 1.2 or aux < promedio_regiones[indice_region] * 0.8:
                        Semilla += 1
                        random.seed(Semilla)
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
    return Casos_nuevos, Casos_reales


def solucion(CS, SS, indice_region, region, idice_fin_entrenamiento, cantidad_dias_entrenamiento, cantidad_dias_proyectados, promedio_regiones):
    Indice_de_saturacion, mediana = indice_saturacion(CS, SS, indice_region, region, idice_fin_entrenamiento, cantidad_dias_entrenamiento)
    x=proyeccion(CS, SS, indice_region, region, cantidad_dias_proyectados, mediana, Indice_de_saturacion, idice_fin_entrenamiento, promedio_regiones)
    return x



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
print(CS_out[Indice_fecha_fin_entrenamiento - 1][0]) # Fecha de inicio de entrenamiento
print(Indice_fecha_fin_entrenamiento)

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

#print(x)

prueba=read_csv("https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto58/Camas_UCI_diarias.csv")
prueba_out=prueba.values
date=prueba.columns
#print(date[17+896]) #913
#print(prueba_out[0][913])
i=913
# camasCovid=prueba_out[17][913]
# camas=prueba_out[0][913]
# print("estas con las camas "+str(prueba_out[0][913]))
# camasTotales=int(camas)
# while True:
#     casos=x[0]
#     aux=((camasCovid*casos)/((1-(camasCovid/camasTotales))*camasTotales))
#     if 0.9>= aux:
#         break
#     camasTotales=camasTotales+1

# print(camasTotales)
j=0
fecha=[]
camasA=[]
indice_sat=[]
camas_reales=[]
region=14
saturacion_real=[]
indice=913
#Casos reales
x,casos_reales=solucion(CS_out, SS_out, region, Regiones, Indice_fecha_fin_entrenamiento, Cantidad_dias_entrenamiento, Cantidad_dias_proyectados, promedio_regiones)
while j<len(x):
    fecha.append(date[913+j])
    j=j+1
i=0
while i<len(x):
    satu=sat(prueba_out,casos_reales,indice,region,i)
    saturacion_real.append(satu)
    camas_reales.append(int(prueba_out[region][913+i]))
    aux,camas=CalculoCamas(prueba_out,x,indice,region,i)
    indice_sat.append(aux)
    camasA.append(int(camas))
    i=i+1
    indice=indice+1
# print(camasA)
# print(indice_sat)
# print(fecha)

plt.xlabel('Fecha')
plt.ylabel('Examenes realizados')
plt.title('Examenes PCR en la region de Magallanes')
#y = [0.0, 0.0, 0.0009999275207519531, 0.0, 0.008999824523925781, 0.003998517990112305, 0.012002229690551758, 0.006997346878051758, 0.041018009185791016, 0.017998933792114258, 0.39300012588500977, 0.3229970932006836, 2.677995443344116, 1.6149992942810059, 13.948997259140015, 0.9549975395202637, 83.28399729728699, 3.9170007705688477, 873.6389403343201]
#fech=[datetime.strptime(date, "%Y-%m-%d").date() for date in fecha]

#x = [4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22]
plt.plot(fecha, x,label='Casos proyectados')
plt.plot(fecha, casos_reales,label='Casos reales')
plt.legend()
plt.gcf().set_size_inches(18, 7)
plt.grid(True, linestyle="-.",c="gray")
plt.savefig('casosVSmagallanes.png')
plt.show()

# x = np.linspace(0, 2, 100)
# #Generamos una grafica lineal para una recta en X
# plt.plot(x, x, label='linear')
# #Generamos otra grafica lineal para una X cuadratica
# plt.plot(x, x**2, label='quadratic')
# #Generamos una grafica lineas para una X Cubica
# plt.plot(x, x**3, label='cubic')
# #Agregamos las etiquetas y añadimos una leyenda.
# plt.xlabel('Indice de saturacion')
# plt.ylabel('Fecha')
# plt.title("Indice de saturacion en la region Arica y Parinacota")
# plt.legend()
# # plt.savefig('grafica_lineal.png')
# plt.show()