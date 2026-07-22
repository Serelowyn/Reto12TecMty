# -------------- importaciones

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# -------------- fin de las importaciones

# 1. se cargan los dos csv del reto de autotransporte federal de turismo
df_entidades = pd.read_csv(r"C:\Users\sasor\Desktop\Tec de mty\5. Analisis de datos\1. Visualizacion de analisis descriptivo\Proyecto\reto_3.csv", encoding="latin-1")
df_meses = pd.read_csv(r"C:\Users\sasor\Desktop\Tec de mty\5. Analisis de datos\1. Visualizacion de analisis descriptivo\Proyecto\reto_1.csv", encoding="latin-1")

# 2. se revisan datos generales de los dataframe para verificar que todo este en orden y limpio, de no ser asi, se haran cambios

print(df_entidades.head())
print(df_entidades.shape)
print(df_entidades.dtypes)
print(df_entidades.info())
print(df_entidades.columns)

print(df_meses.head())
print(df_meses.shape)
print(df_meses.dtypes)
print(df_meses.info())
print(df_meses.columns)

# la columna Personal trae un espacio de mas al final, se quita
df_meses = df_meses.rename(columns={"Personal ": "Personal"})

print(df_entidades.isnull().sum())
print(df_meses.isnull().sum())
"""no se requieren cambios de null"""

columnas_vehiculos = ["Automovil", "Autobus", "Camioneta"]

# 3. grafica de caja de bigotes

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_entidades[columnas_vehiculos])
plt.title("distribucion del num de vehiculos por clase")
plt.xlabel("clase de vehiculo")
plt.ylabel("unidades")
plt.show()

# 4. histograma: automoviles por entidad

plt.figure(figsize=(10, 6))
sns.histplot(
    data=df_entidades,
    x="Automovil",
    bins=8,
    kde=False
)

plt.title("distribucion del numero de automoviles por cada entidad")
plt.xlabel("num de automoviles")
plt.ylabel("frecuencia")
plt.show()

# kdeplot: densidad de pasajeros mensuales

plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=df_meses,
    x="Pasajeros",
    fill=True
)

plt.title("distribucion kde pasajeros mensuales")
plt.xlabel("pasajeros")
plt.ylabel("densidad")
plt.show()
