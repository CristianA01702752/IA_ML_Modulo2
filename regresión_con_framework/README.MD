# Momento de Retroalimentación: Módulo 2 Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución. (Portafolio Implementación)
Cristian Espinosa Díaz A01702752
Archivos: linear_framework.py, FastFoodNutrition.csv

Notas: Se necesita la instalación de Python 3 y librerías numpy, pandas, matplotlib, seaborn para la ejecución del código.

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

# Regresión utilizada

Se escogió utilizar regresión lineal debido a que el objetivo del modelo es predecir el comportamiento sobre la cantidad de calorías con respecto a las Grasas Saturadas y Grasas Totales en alimentos de comida rápida. La elección de las variables se debe a que una fuente principal de calorías en alimentos suele ser la grasa. A su vez, otro factor relevante a considerar es que en el código se realiza una matriz de correlación para establecer que hay correlaciones altas ya que el modelo sin framework sobre las proteínas no es tan significativo. 

# Gráficos y aspectos relevantes a considerar

Dentro de este entregable se realiza una ligera limpieza del dataset ya que se detectó un porcentaje ligero sobre datos vacíos o con información corrompida.

Posterior a eso, se realiza una división del dataset para entrenamiento y prueba (train y test). En este, el 80% de los datos es utilizado en el entrenamiento y el 20% para las pruebas/predicciones.

Al finalizar la ejecución, el código imprime lo siguiente:

* Coeficientes que mutiplicaría a las variables independientes (total fat y saturated fat)
* MSE de train: Se imprime el último valor de error cuadrático en la fase de entrenamiento.
* R-squared de train: Se imprime el valor de la r-cuadrada, este valor es utlizado para dar una idea sobre que tan bien el modelo explica la variabilidad en la variable dependiente. Es decir, la precisión del modelo.
* MSE de test: Se imprime el último valor de error cuadrático en la fase de entrenamiento.
* R-squared de test: Se imprime el valor de la r-cuadrada, este valor es utlizado para dar una idea sobre que tan bien el modelo explica la variabilidad en la variable dependiente. Es decir, la precisión del modelo.
* Cross validation: El valor de r-cuadrada después de realizar cross validation usando los datos de train. Cross validation es una técnica que se utiliza para evaluar qué tan bien funciona un modelo de machine learning. En el código se realiza una cross validation utilizando 10 pliegues y calcula el promedio del r-cuadrada como una medida del rendimiento del modelo. Esto ayuda a evaluar qué tan bien se espera que el modelo funcione en datos nuevos y no vistos.
* Gráfico "Train real data vs Train predicted data": Este gráfico compara visualmente los valores reales de "Calories from fat" con los valores predichos por un modelo en el conjunto de entrenamiento.
* Gráfico "Test real data vs Test predicted data": Este gráfico compara visualmente los valores reales de "Calories from fat" con los valores predichos por un modelo en el conjunto de pruebas.
