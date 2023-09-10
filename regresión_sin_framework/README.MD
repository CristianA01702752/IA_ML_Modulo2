# Momento de Retroalimentación: Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación)
Cristian Espinosa Díaz A01702752
Archivos: linear_without_fr.py, FastFoodNutrition.csv

Notas: Se necesita la instalación de Python 3 y librerías numpy, pandas y matplotlib para la ejecución del código.

La ejecución del código puede tardar algunos segundos.

# Dataset
Para esta entrega se utiliza la primera versión del dataset Fast Food Nutrition Menu (https://www.kaggle.com/datasets/joebeachcapital/fast-food/versions/1), el cual contiene registros sobre valores nutritionales, micronutrientes y calories de 6 restaurantes de comida rápida (McDonald's, Burger King, Wendy's, Kentucky Fried Chicken, Taco Bell, Pizza Hut).

El dataset posee 14 columnas sobre cada platillo registrado:

* Company: El restaurante donde se localiza el platillo
* Item: Nombre oficial del platillo
* Calories: Cantidad de calorías provenientes de todas las fuentes (unidad de medición "cal") 
* Calories fromFat: Cantidad de calorías provenientes de grasa(unidad de medición "cal") 
* Total Fat(g): Cantidad de grasas totales(unidad de medición "gramos") 
* Saturated Fat(g): Cantidad de grasas saturadas(unidad de medición "gramos") 
* Trans Fat(g): Cantidad de grasas trans(unidad de medición "gramos") 
* Cholesterol(mg): Cantidad de colesterol(unidad de medición "miligramos") 
* Sodium (mg): Cantidad de sodio (unidad de medición "gramos") 
* Carbs(g): Cantidad de carbohidratos (unidad de medición "gramos") 

# Algoritmo utilizado
Para este entregable se implementó el algoritmo de Gradient Descent con regresión lineal.

Se escogió utilizar regresión lineal debido a que el objetivo del modelo es predecir el comportamiento sobre la cantidad de proteínas con respecto a las Grasas Saturadas en alimentos de comida rápida. La elección de las variables se debe a que las proteínas son encontradas con mayor frecuencia en productos de origen animal y un gran porcentaje de los menus de distintos restaurantes de comida rápida poseen platillos con dichos productos. A su vez, la grasa saturada tiene mayor frecuencia en alimentos de estas franquicias por el uso de aceites y otros elementos en su elaboración. 

# Gráficos y aspectos relevantes a considerar

Dentro de este entregable se realiza una ligera limpieza del dataset ya que se detectó un porcentaje ligero sobre datos vacíos o con información corrompida.

Posterior a eso, se realiza una división del dataset para entrenamiento y prueba (train y test). En este, el 80% de los datos es utilizado en el entrenamiento y el 20% para las pruebas/predicciones.

Al finalizar la ejecución, el código imprime lo siguiente:

* Ecuación final (los valores finales después del entrenamiento para una ecuación multivariable de y = m1*x1 + m2*x2 + b)
* MSE con datos de entrenamiento: Se imprime el último valor de error cuadrático en la fase de entrenamiento.
* R-squared: Se imprime el valor de la r-cuadrada, este valor es utlizado para dar una idea sobre que tan bien el modelo explica la variabilidad en la variable dependiente. Es decir, la precisión del modelo.
* MSE con datos de prueba: Se imprime el último valor de error cuadrático en la fase de prueba.
* Gráfico "Modelo": Este gráfico se ilustra la relación entre "Grasas Saturadas" en el eje x y su influencia conjunta en "Proteína" en el eje y. La línea roja en el gráfico representa la línea de regresión, que refleja la relación lineal aprendida por el modelo entre estas variables. 
* Gráfico "Evolución del Error Cuadrático Medio (MSE)": Este gráfico muestra los cambios que sufre el MSE conforme las iteraciones en el entrenamiento.

# Parámetros
Los hiperparámetros se encuentran definidos en el código:

Alfa o learning rate: 0.0001 (se puede localizar en la línea 35)
Epoch (Épocas): 250 (se puede localizar en la línea 38)

Estos parámetros se asignaron después de pruebas con distintos valores para el entrenamiento, obteniendo un buen resultado sobre los datos de prueba.
