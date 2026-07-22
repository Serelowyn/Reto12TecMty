# -------------- importaciones

import pandas as pd
 
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

# se revisan nulos por si hace falta rellenar algo, en este caso no hay nulos
print(df_entidades.isnull().sum())
print(df_meses.isnull().sum())
"""no se requieren cambios de null"""

columnas_vehiculos = ["Automovil", "Autobus", "Camioneta"]