import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Lectura del dataset
columns = ["Company","Item","Calories","Calories from","Total Fat","Saturated Fat","Trans Fat","Cholesterol","Sodium","Carbs","Fiber","Sugars","Protein","Weight"]
df = pd.read_csv('FastFoodNutritionMenu.csv',names=columns)

"""
 Asignación de la variable dependiente e independiente
 Para este modelo la columna "proteína" es la dependiente
 Las variables independientes serán carbohidratos y grasas saturadas
"""

# Eliminación las filas con valores faltantes en las variables de interes 
df['Protein'] = pd.to_numeric(df['Protein'], errors='coerce')
df['Saturated Fat'] = pd.to_numeric(df['Saturated Fat'], errors='coerce')
df = df.dropna(subset=['Protein', 'Saturated Fat', 'Carbs'])

# Eliminación de las filas donde los datos sean iguales a 0
df = df.loc[abs(df['Protein']) > 0]
df = df.loc[abs(df['Saturated Fat']) > 0]

Y = df['Protein'].to_numpy()
X = df[['Saturated Fat']].to_numpy()

# Dividimos los datos en 80% para entrenar el modelo y 20% para el testeo del modelo
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

m = 0  # Pendiente para la variable independiente
b = 0 # Término constante o intercepto

# Learning rate, el valor que más se adaptó al modelo fue 0.0001
learning_rate = 0.0001

# Cantidad de iteraciones/épocas del modelo. El valor que más se adaptó al modelo fue 250
epoch = 250

# Longitud del dataset
n = len(X_train)

# Arreglo para almacenar la evolución del mse a lo largo de las iteraciones
mse_values = []


def mean_square(y, m, x, b, n):
    acum = 0
    for i in range(n):
        diff = y[i] - (m * x[i] + b)
        acum += diff ** 2
    return acum / n

# Algoritmo de gradient descent
for i in range(epoch):
    # Se obtiene la hipótesis/predicción
    Y_train_pred = (m * X_train[:, 0]) + b
    error = mean_square(Y_train, m, X_train[:, 0], b, len(X_train))
    mse_values.append(error)
    
    # Se calculan las derivadas usando la sumatoria de las diferencias entre la predicción y el valor real (errores)
    Deriv_m = (-2 / len(X_train)) * sum(X_train[:, 0] * (Y_train - Y_train_pred))
    Deriv_b = (-2 / len(X_train)) * sum(Y_train - Y_train_pred)
    
    # Se actualizan los valores de m1, m2 y b
    m = m - learning_rate * Deriv_m
    b = b - learning_rate * Deriv_b

print("Ecuación Final:")
print ("y =", m, "* x +", b)

# Evaluación del modelo en los datos de entrenamiento
print()
print("MSE con datos de entrenamiento:", error)

# Calcular el coeficiente de determinación (R-squared) en los datos de entrenamiento
SST_train = np.sum((Y_train - np.mean(Y_train)) ** 2)
SSE_train = np.sum((Y_train - Y_train_pred) ** 2)
R_squared_train = 1 - (SSE_train / SST_train)
print("R-squared:", R_squared_train)

# Evaluación del modelo en los datos de prueba
Y_test_pred = (m * X_test[:, 0]) + b

# Calcular el MSE en los datos de prueba
MSE_test = mean_square(Y_test, m, X_test[:, 0], b, len(X_test))
print()
print("MSE con datos de prueba:", MSE_test)

# Graficamos los resultados
plt.scatter(X[:, 0], Y, color='blue')  # Gráfico para la variable independiente 'x1'
plt.plot(X_train[:, 0], Y_train_pred, color='red', label='Regresión Lineal')
plt.xlabel('Saturated Fat')
plt.ylabel('Protein')
plt.legend()
plt.show()

# Graficamos la evolución del MSE en período de train
plt.plot(range(epoch), mse_values)
plt.xlabel('Iteración')
plt.ylabel('MSE')
plt.title('Evolución del Error Cuadrático Medio (MSE)')
plt.show()