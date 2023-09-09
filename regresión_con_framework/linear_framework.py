import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

columns = ["Company","Item","Total Calories","Calories from fat","Total Fat","Saturated Fat","Trans Fat","Cholesterol","Sodium","Carbs","Fiber","Sugars","Protein","Weight"]

# Leer el archivo CSV
df = pd.read_csv('FastFoodNutritionMenu.csv', names=columns)

# Convertir todas las columnas relevantes a tipo numérico
df_temp = df[["Total Calories","Calories from fat","Total Fat","Saturated Fat","Trans Fat","Cholesterol","Sodium","Carbs","Fiber","Sugars","Protein"]]
for column in df_temp.columns:
    df_temp[column] = pd.to_numeric(df_temp[column], errors='coerce')
# Eliminar filas con valores nulos en las columnas relevantes
df_temp = df_temp.dropna()

# Calcular la matriz de correlación
corr_matrix = df_temp.corr()

# Crear y mostrar el mapa de calor
plt.figure(figsize=(10, 6))
sn.heatmap(corr_matrix, annot=True, linewidths=0.3, fmt='0.2f')
plt.show()

# Eliminación las filas con valores faltantes en las variables de interes 
df['Calories from fat'] = pd.to_numeric(df['Calories from fat'], errors='coerce')
df['Total Fat'] = pd.to_numeric(df['Total Fat'], errors='coerce')
df['Saturated Fat'] = pd.to_numeric(df['Saturated Fat'], errors='coerce')
df = df.dropna(subset=['Calories from fat', 'Total Fat', 'Saturated Fat'])

"""
 Asignacion de la variable dependiente e independiente
 Para este modelo la columna "Calories from fat" es la dependiente
 Las variables independientes serán grasas totales (Total fat) 
 y grasas saturadas (Saturated fat)
"""
Y = df['Calories from fat']
X = df[['Total Fat', 'Saturated Fat']]

# Dividimos los datos en 80% para entrenar el modelo y 20% para el testeo del modelo
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
model = LinearRegression(fit_intercept=True)

# Entrenamiento del modelo
model.fit(X_train, Y_train)

# Valores de la ecuación del modelo
coefficients = model.coef_
print("Coeficiente para Total Fat:", coefficients[0])
print("Coeficiente para Saturated Fat:", coefficients[1])
print ("Intersección: ", model.intercept_)

# Predicciones del modelo con los datos de entrenamiento (train)
Y_predict_train = model.predict(X_train)
# Mean Square Error de la fase de entrenamiento 
# de los datos que predice el modelo contra los datos reales
print ("MSE de train: ", mean_squared_error(Y_train, Y_predict_train ))
# Precisión del modelo en la fase de entrenamiento
print ("R-squared de train: ", r2_score(Y_train, Y_predict_train ))

# Predicciones del modelo con los datos de prueba (test)
Y_predict_test= model.predict(X_test)
# Mean Square Error de la fase de prueba
# de los datos que predice el modelo contra los datos reales
print ("MSE de test: ", mean_squared_error(Y_test, Y_predict_test))
# Precisión del modelo
print ("R-squared de test: ", r2_score(Y_test, Y_predict_test))

# Cross Validation del modelo
Cross_val = abs(cross_val_score(LinearRegression(), X_train, Y_train, cv=10, scoring = "r2").mean())
print ("Cross validation: ", Cross_val)


plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
sn.scatterplot(x=range(len(Y_train)), y=Y_train, label='y-train real data')
sn.scatterplot(x=range(len(Y_train)), y=Y_predict_train, label='y-train predicted data by the model')

plt.ylabel("Calories from fat")
plt.xlabel("Total Fat and Saturated Fat")
plt.legend()
plt.title("Train real data vs Train predicted data")
plt.show()

plt.figure(figsize=(10, 10))

plt.subplot(1, 2, 1)
sn.scatterplot(x=range(len(Y_test)), y=Y_test, label='y-test real data')
sn.scatterplot(x=range(len(Y_test)), y=Y_predict_test, label='y-test predicted data by the model')

plt.ylabel("Calories from fat")
plt.xlabel("Total Fat and Saturated Fat")
plt.legend()
plt.title("Test real data vs Test predicted data")

plt.show()
