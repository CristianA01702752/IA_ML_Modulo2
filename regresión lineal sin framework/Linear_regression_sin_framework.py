import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

columns = ["Company","Item","Calories","Calories from","Total Fat","Saturated Fat","Trans Fat","Cholesterol","Sodium","Carbs","Fiber","Sugars","Protein","Weight"]

# I read the dataset and store it in a dataframe from pandas
df = pd.read_csv('FastFoodNutritionMenu.csv',names=columns)

# Elimina las filas con valores faltantes en 'Protein' o 'Saturated Fat'
df['Protein'] = pd.to_numeric(df['Protein'], errors='coerce')
df['Carbs'] = pd.to_numeric(df['Carbs'], errors='coerce')
df['Saturated Fat'] = pd.to_numeric(df['Saturated Fat'], errors='coerce')
df = df.dropna(subset=['Protein', 'Saturated Fat', 'Carbs'])

# Elimina las filas donde los datos sean iguales a 0
df = df.loc[abs(df['Protein']) > 0]
df = df.loc[abs(df['Carbs']) > 0]
df = df.loc[abs(df['Saturated Fat']) > 0]

"""
 Asignacion de la variable dependiente e independiente
 Para este modelo la columna "proteína" es la dependiente
 Las variables independientes será carbohidratos y grasas saturadas
"""
Y = df['Protein'].to_numpy()
X = df[['Carbs', 'Saturated Fat']].to_numpy()

# Dividimos los datos en 80% para entrenar el modelo y 20% para el testeo del modelo
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

m1 = 0  # Pendiente para la primera variable independiente
m2 = 0  # Pendiente para la segunda variable independiente
b = 0 # Término constante o intercepto

# Learning rate, el valor que más se adaptó al modelo fue 0.000001
learning_rate = 0.000001

# Cantidad de iteraciones/épocas del modelo el valor que más se adaptó al modelo fue 500
epoch = 500

# Longitud del dataset
n = len(X)

# Arreglo para almacenar la evolución del mse a lo largo de las iteraciones
mse_values = []


def mean_square (y, m1, m2, x, b, n):
  acum = 0
  for i in range (0,n):
    diff = y[i] - (m1 * x[i][0] + m2 * x[i][1] + b)
    acum += diff**2
  acum = acum / n
  return acum

# Algoritmo de gradient descent
for i in range(epoch):
  
    # Se obtiene la hipótesis/predicción
    Y_train_pred = (m1 * X_train[:, 0] + m2 * X_train[:, 1]) + b
    error = mean_square(Y_train, m1, m2, X_train, b, len(X_train))
    mse_values.append(error)
    
    # Se calculan las derivadas usando la sumatoria de las diferencias entre la predicción y el valor real (errores)
    Deriv_m1 = (-2/len(X_train)) * sum(X_train[:, 0] * (Y_train - Y_train_pred))
    Deriv_m2 = (-2/len(X_train)) * sum(X_train[:, 1] * (Y_train - Y_train_pred))
    Deriv_b = (-2/len(X_train)) * sum(Y_train - Y_train_pred)
    
    # Se actualizan los valores de m1, m2 y b
    m1 = m1 - learning_rate * Deriv_m1
    m2 = m2 - learning_rate * Deriv_m2
    b = b - learning_rate * Deriv_b

print("Ecuación Final:")
print ("y =", m1, "* x1 +", m2, "* x2 +", b)

# Evaluación del modelo en los datos de entrenamiento
print()
print("MSE con datos de entrenamiento:", error)

# Calcular el coeficiente de determinación (R-squared) en los datos de entrenamiento
SST_train = np.sum((Y_train - np.mean(Y_train)) ** 2)
SSE_train = np.sum((Y_train - Y_train_pred) ** 2)
R_squared_train = 1 - (SSE_train / SST_train)
print("R-squared con datos de entrenamiento:", R_squared_train)

# Evaluación del modelo en los datos de prueba
Y_test_pred = (m1 * X_test[:, 0] + m2 * X_test[:, 1]) + b

# Calcular el MSE en los datos de prueba
MSE_test = mean_square(Y_test, m1, m2, X_test, b, len(X_test))
print()
print("MSE con datos de prueba:", MSE_test)

# Calcular el coeficiente de determinación (R-squared) en los datos de entrenamiento
SST_test = np.sum((Y_test - np.mean(Y_test)) ** 2)
SSE_test = np.sum((Y_test - Y_test_pred) ** 2)
SST_test = np.sum((Y_test - np.mean(Y_test)) ** 2)

R_squared_test = 1 - (SSE_test / SST_test)
print("R-squared con datos de prueba:", R_squared_test)

# Graficamos los resultados
plt.scatter(X[:, 0], Y)  # Gráfico para la primera variable independiente
plt.scatter(X[:, 1], Y)  # Gráfico para la segunda variable independiente
plt.plot(X[:, 0], (m1 * X[:, 0] + m2 * X[:, 1]) + b, color='red', label='Regresión Lineal')
plt.xlabel('Carbs and Saturated Fats')
plt.ylabel('Protein')
plt.legend()
plt.show()

# Graficamos la evolución del MSE en período de train
plt.plot(range(epoch), mse_values)
plt.xlabel('Iteración')
plt.ylabel('MSE')
plt.title('Evolución del Error Cuadrático Medio (MSE)')
plt.show()