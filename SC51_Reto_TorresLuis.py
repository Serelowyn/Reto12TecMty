# -------------- importaciones

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

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

# 3. grafica de caja de bigotes: mediana de autobuses por entidad

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_entidades, y="Autobus")
plt.title("caja de bigotes: autobuses por entidad")
plt.xlabel("autobus")
plt.ylabel("unidades")
plt.show()

mediana_autobus = df_entidades["Autobus"].median()
print(f"mediana de autobuses entre entidades: {mediana_autobus}")

# 4. grafica tridimensional: automovil, autobus y camioneta por entidad

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    df_entidades["Automovil"],
    df_entidades["Autobus"],
    df_entidades["Camioneta"],
    c="red"
)
ax.set_xlabel("automovil")
ax.set_ylabel("autobus")
ax.set_zlabel("camioneta")
ax.set_title("relacion entre automovil, autobus y camioneta por entidad")
plt.show()

# 5. grafica residual: automovil (dependiente) vs autobus y camioneta (independientes)

X_multiple = df_entidades[["Autobus", "Camioneta"]]
Y_multiple = df_entidades["Automovil"]

modelo_multiple = LinearRegression()
modelo_multiple.fit(X_multiple, Y_multiple)
Y_multiple_pred = modelo_multiple.predict(X_multiple)
residuos_multiple = Y_multiple - Y_multiple_pred

print("intercepto:", modelo_multiple.intercept_)
print("coeficientes (autobus, camioneta):", modelo_multiple.coef_)

plt.figure(figsize=(10, 6))
plt.scatter(Y_multiple_pred, residuos_multiple, color="red", s=70)
plt.axhline(y=0, color="gray", linestyle="--")
plt.title("grafica residual: automovil ~ autobus + camioneta")
plt.xlabel("automoviles pronosticados")
plt.ylabel("residuo")
plt.show()

# 6. relacion de ajuste: automovil vs autobus

slope, intercept, r_value, p_value, std_err = linregress(df_entidades["Autobus"], df_entidades["Automovil"])

print("ecuacion de la recta:")
print(f"automovil = {intercept:.4f} + ({slope:.4f}) * autobus")
print(f"r = {r_value:.4f}")
print(f"r^2 = {r_value**2:.4f}")
print(f"p-value = {p_value:.4f}")

plt.figure(figsize=(10, 6))
sns.regplot(
    data=df_entidades,
    x="Autobus",
    y="Automovil",
    scatter_kws={"s": 70},
    line_kws={"color": "red"}
)
plt.text(
    0.05,
    0.95,
    f"automovil = {intercept:.2f} + ({slope:.2f}) * autobus\nR² = {r_value**2:.2f}",
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white")
)

plt.title("relacion de ajuste: automovil vs autobus")
plt.xlabel("autobus")
plt.ylabel("automovil")
plt.show()

if abs(r_value) >= 0.7:
    fuerza = "fuerte"
elif abs(r_value) >= 0.3:
    fuerza = "moderada"
else:
    fuerza = "debil o nula"
signo = "positiva" if r_value > 0 else "negativa"
print(f"existe una relacion {signo} {fuerza} entre automovil y autobus")

# matriz de correlacion de automovil y autobus (las variables de este punto)
print("matriz de correlacion")
print(df_entidades[["Automovil", "Autobus"]].corr())

# 7. kdeplot: relacion entre automoviles y autobuses de las primeras 6 entidades

primeras_6 = df_entidades.head(6)
print(primeras_6)

plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=primeras_6,
    x="Automovil",
    y="Autobus",
    fill=True,
    cmap="Blues"
)
plt.title("kdeplot: relacion automovil vs autobus (primeras 6 entidades)")
plt.xlabel("automovil")
plt.ylabel("autobus")
plt.show()

# 8. histograma: personal vs meses, solo de enero a junio

meses_orden = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio"]
primeros_6_meses = df_meses[df_meses["Meses"].isin(meses_orden)].copy()
primeros_6_meses["Meses"] = pd.Categorical(primeros_6_meses["Meses"], categories=meses_orden, ordered=True)
primeros_6_meses = primeros_6_meses.sort_values("Meses")
print(primeros_6_meses)

plt.figure(figsize=(10, 6))
sns.histplot(
    data=primeros_6_meses,
    x="Meses",
    weights="Personal",
    shrink=0.8
)
plt.title("histograma: personal por mes (enero a junio)")
plt.xlabel("mes")
plt.ylabel("personal")
plt.show()

# 9. grafica de pastel: automoviles de las primeras 6 entidades

plt.figure(figsize=(8, 8))
plt.pie(
    primeras_6["Automovil"],
    labels=primeras_6["Entidad"],
    autopct="%1.1f%%",
    startangle=90
)
plt.title("proporcion de automoviles - primeras 6 entidades")
plt.show()

# 10. grafica de dispersion: automovil vs autobus

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_entidades,
    x="Autobus",
    y="Automovil",
    s=80
)
plt.title("dispersion: automovil vs autobus")
plt.xlabel("autobus")
plt.ylabel("automovil")
plt.show()

print(f"la dispersion muestra una relacion {signo} {fuerza} entre automovil y autobus")