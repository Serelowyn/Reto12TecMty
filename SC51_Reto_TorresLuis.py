# -------------- importaciones

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import linregress

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

# 5. kdeplot: densidad de pasajeros mensuales

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

# 6. grafica de dispersion:

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_meses,
    x="Autobuses",
    y="Personal",
    s=80
)

plt.title("relacion entre autobuses y personal")
plt.xlabel("autobuses")
plt.ylabel("personal")
plt.show()

# 7. relacion de ajuste: pasajeros vs kilometros recorridos

# ajuste, regresion lineal
slope, intercept, r_value, p_value, std_err = linregress(df_meses["Pasajeros"], df_meses["Kilómetros"])

# ecuacion de la recta
print("ecuacion de la recta:")
print(f"kilometros = {intercept:.4f} + ({slope:.4f}) * pasajeros")
print(f"r = {r_value:.4f}")
print(f"r^2 = {r_value**2:.4f}")
print(f"p-value = {p_value:.4f}")

plt.figure(figsize=(10, 6))
sns.regplot(
    data=df_meses,
    x="Pasajeros",
    y="Kilómetros",
    scatter_kws={"s": 70},
    line_kws={"color": "red"}
)

# se agrega ecuacion a la grafica en texto
plt.text(
    0.05,
    0.95,
    f"kilometros = {intercept:.2f} + ({slope:.2f}) * pasajeros\nR² = {r_value**2:.2f}",
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white")
)

plt.title("relacion de ajuste: pasajeros vs kilometros recorridos")
plt.xlabel("pasajeros")
plt.ylabel("kilometros")
plt.show()

# matriz de correlacion de las variables numericas de df_meses
print("matriz de correlacion")
print(df_meses[["Autobuses", "Personal", "Pasajeros", "Kilómetros"]].corr())

# la relacinon puede ser fuerte, moderada o debil segun el valor de r
if abs(r_value) >= 0.7:
    fuerza = "fuerte"
elif abs(r_value) >= 0.3:
    fuerza = "moderada"
else:
    fuerza = "debil o nula"
signo = "positiva" if r_value > 0 else "negativa"
print(f"existe una relacion {signo} {fuerza} entre pasajeros y kilometros recorridos")
